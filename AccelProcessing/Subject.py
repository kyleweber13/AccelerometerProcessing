import pyedflib
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import AccelProcessing.Accelerometer as Accelerometer
import AccelProcessing.Filtering as Filtering
from nwfiles.pipeline import nwdata as nwdata


class Subject:

    def __init__(self, subj_id=None, wrist_filepath=None, ankle_filepath=None, processed_filepath=None,
                 load_raw=True, from_processed=False, output_dir="", epoch_len=15, cutpoints="Powell",
                 write_epoched_data=False, write_intensity_data=False, overwrite_output=False):
        """Class to read in EDF-formatted wrist and ankle accelerometer files.

           N.B.:
               -SVM values are in Gs while AVM values are in mG!!!
               -Fraysse cutpoints are for categorizing activity into sedentary, light, or MVPA while Powell cutpoints
                are for sedentary, light, moderate, and vigorous. When using Fraysse cutpoints, daily vigorous activity
                will be 0s even though it was not calculated. MVPA is calculated with either cutpoint set.

        :argument
        -subj_id: used for file naming, str
            -format example: "OND07_WTL_3013", or "007"

        -wrist_filepath: full pathway to wrist .edf file
            -Include {} where subj_id gets inserted
            -If no file, pass None
        -ankle_filepath: full pathway to ankle .edf file
            -Include {} where subj_id gets inserted
            -If no file, pass None

        -load_raw: whether to load raw data; boolean
        -epoch_len: epoch length in seconds, int

        -processed_filepath: full pathway to .csv file created using Subject.create_epoch_df() (1-second epochs)
                             OR existing csv file
            -Needs to include {} where the subject ID should be
                -E.g.: {}_Epoch1_Accelerometer.csv
        -from_processed: whether to load file specified using processed_filepath; boolean

        -cutpoints: str for which cutpoints to use
            -"Powell" for Powell et al. 2017 or "Fraysse" for Fraysse et al. 2021

        -output_dir: full pathway to where files get written
        -write_epoched_data: whether to write df_epoch to .csv; boolean
            -One row for each epoch; contains WristSVM, WristAVM, AnkleSVM, AnkleAVM
        -write_intensity_data: whether to write df_daily and activity_totals to .csv's; boolean
        -overwrite_output: whether to automatically overwrite existing df_epoch file; boolean
            -If False, user will be prompted to manually overwrite existing file.
        """

        self.subj_id = subj_id
        self.wrist_exists = False
        self.ankle_exists = False

        if wrist_filepath is not None:
            self.wrist_filepath = wrist_filepath.format(subj_id)
            self.wrist_exists = os.path.exists(self.wrist_filepath)
        if wrist_filepath is None:
            self.wrist_filepath = None

        if ankle_filepath is not None:
            self.ankle_filepath = ankle_filepath.format(subj_id)
            self.ankle_exists = os.path.exists(self.ankle_filepath)
        if ankle_filepath is None:
            self.ankle_filepath = None

        self.processed_filepath = processed_filepath.format(subj_id)
        self.output_dir = output_dir
        self.load_raw = load_raw
        self.from_processed = from_processed
        self.epoch_len = epoch_len
        self.cutpoints = cutpoints
        self.write_epoched = write_epoched_data
        self.write_intensity_data = write_intensity_data
        self.overwrite_output = overwrite_output

        self.activity_totals = {"Sedentary": 0, "Light": 0, "Moderate": 0, "Vigorous": 0, "MVPA": 0}
        self.df_daily = pd.DataFrame(columns=["Date", "Sedentary", "Light", "Moderate", "Vigorous", "MVPA"])

        # ================================================== RUNS METHODS =============================================
        self.print_summary()

        if self.load_raw:
            self.wrist_offset, self.ankle_offset = self.sync_starts()

            self.wrist, self.cutpoint_dict = self.create_wrist_obj()

            self.wrist_svm, self.wrist_avm = self.epoch_accel(acc_type="wrist",
                                                              fs=self.wrist.fs if self.wrist_exists else 1,
                                                              vm_data=self.wrist.accel_vm if self.wrist_exists else [],
                                                              epoch_len=1)

            if not self.wrist_exists:
                self.wrist = None
                self.cutpoint_dict = None
                self.wrist_svm = None
                self.wrist_avm = None

            self.ankle = self.create_ankle_obj()

            self.ankle_svm, self.ankle_avm = self.epoch_accel(acc_type="ankle",
                                                              fs=self.ankle.fs if self.ankle_exists else 1,
                                                              vm_data=self.ankle.accel_vm if self.ankle_exists else 1,
                                                              epoch_len=1)

            self.df_epoch1s = self.create_epoch_df(epoch_len=1)
            self.df_epoch = self.create_epoch_df(epoch_len=self.epoch_len)
            del self.wrist_svm, self.wrist_avm, self.ankle_svm, self.ankle_avm

        if self.from_processed:
            self.df_epoch1s, self.df_epoch = self.import_processed_df(epoch_len=self.epoch_len)

        if (self.load_raw and self.wrist_exists) or (not self.wrist_exists and self.from_processed):
            self.calculate_wrist_intensity()

        if not self.wrist_exists and not self.from_processed:
            print("\nSkipping wrist intensity calculation (no file given).")
            self.df_epoch["WristIntensity"] = [None for i in range(self.df_epoch.shape[0])]

        if not self.wrist_exists and self.from_processed:
            print("\nSkipping wrist intensity calculation (unknown sample rate).")
            self.df_epoch["WristIntensity"] = [None for i in range(self.df_epoch.shape[0])]

        if self.write_epoched:
            self.write_epoched_df()

        print("\n====================================================================================================")
        print("Processing complete.")

    def print_summary(self):
        """Prints summary of what data will be read in."""

        print("======================================================================================================")
        print("\nData import summary:")

        if self.load_raw:
            print("\n-Raw data will be imported.")

            if self.wrist_exists:
                print("-Importing wrist file: {}".format(self.wrist_filepath))
            if not self.wrist_exists:
                print("-No wrist file will be imported.")

            if self.ankle_exists:
                print("-Importing ankle file: {}".format(self.ankle_filepath))
            if not self.ankle_exists:
                print("-No ankle file will be imported.")

        if not self.load_raw:
            print("\n-Raw data will not be imported.")

        if not self.from_processed:
            print("-Data will not be read from processed.")
        if self.from_processed:
            print("-Data will be read from processed.")

        print()
        print("======================================================================================================")

    @staticmethod
    def check_file(filepath, print_summary=True):
        """Checks EDF header info to retrieve start time and sample rate.
           Used for cropping ankle and wrist accelerometers data.

        :returns
        -start time: timestamp
        -sampling rate: Hz (int)
        """

        if filepath is None:
            return None, None

        if filepath is not None and not os.path.exists(filepath):
            print("Could not find {}.".format(filepath))
            print("Try again.")
            return None, None

        if filepath is not None and os.path.exists(filepath):

            edf_file = pyedflib.EdfReader(filepath)

            duration = edf_file.getFileDuration()
            start_time = edf_file.getStartdatetime()
            end_time = start_time + timedelta(seconds=edf_file.getFileDuration())

            if print_summary:
                print("\n", filepath)
                print("-Sample rate: {}Hz".format(edf_file.getSampleFrequency(0)))
                print("-Start time: ", start_time)
                print("-End time:", end_time)
                print("-Duration: {} hours".format(round(duration / 3600, 2)))

            return start_time, edf_file.getSampleFrequency(0)
    
    def sync_starts(self):
        """Crops ankle/wrist accelerometer file so they start at the same time.

        :returns
        -ankle_offset, wrist_offest: number of samples that one file gets cropped by. Other value will be 0.
        """

        a_start, a_sample = self.check_file(filepath=self.ankle_filepath, print_summary=False)
        w_start, w_sample = self.check_file(filepath=self.wrist_filepath, print_summary=False)

        # Crops start times
        if a_start is not None and w_start is not None:

            print("Cropping data files so they start at the same time...")

            time_offset = (w_start - a_start).total_seconds()

            # If files started at same time
            if time_offset == 0:
                return 0, 0

            if time_offset < 0:
                wrist_offset = abs(int(time_offset * w_sample))

                return wrist_offset, 0

            if time_offset > 0:
                ankle_offset = abs(int(time_offset * a_sample))

                return 0, ankle_offset

        # If only one file given
        if a_start is None or w_start is None:

            print("Only one file was given. Skipping start cropping.")

            return 0, 0

    def create_ankle_obj(self):
        """Creates ankle accelerometer data object.

        :returns
        -ankle object
        """

        print("\n--------------------------------------------- Ankle file --------------------------------------------")

        if self.ankle_exists:
            ankle = nwdata.nwdata()
            ankle.import_edf(file_path=self.ankle_filepath)

            fs = ankle.signal_headers[0]["sample_rate"]
            ankle.fs = fs

            ankle.x = ankle.signals[0][self.ankle_offset:]
            ankle.y = ankle.signals[0][self.ankle_offset:]
            ankle.z = ankle.signals[0][self.ankle_offset:]

            # Calculates gravity-subtracted vector magnitude
            # Negative values become zero
            ankle.accel_vm = np.sqrt(np.square(ankle.signals).sum(axis=0)) - 1
            ankle.accel_vm[ankle.accel_vm < 0] = 0
            del ankle.signals

            ankle.timestamps = pd.date_range(start=ankle.header["startdate"],
                                             freq="{}ms".format(round(1000 / fs, 6)), periods=len(ankle.x))

        if not self.wrist_exists:
            ankle = None

        return ankle

    def create_wrist_obj(self):
        """Creates wrist accelerometer data object.
           Scales accelerometer cutpoints from Powell et al. (2017) to selected epoch length.

        :returns
        -wrist object
        -cutpoints: dictionary
        """

        print("\n--------------------------------------------- Wrist file --------------------------------------------")
        if self.wrist_exists:
            wrist = nwdata.nwdata()
            wrist.import_edf(file_path=self.wrist_filepath)

            fs = wrist.signal_headers[0]["sample_rate"]
            wrist.fs = fs

            wrist.x = wrist.signals[0][self.wrist_offset:]
            wrist.y = wrist.signals[0][self.wrist_offset:]
            wrist.z = wrist.signals[0][self.wrist_offset:]

            # Calculates gravity-subtracted vector magnitude
            # Negative values become zero
            wrist.accel_vm = np.sqrt(np.square(wrist.signals).sum(axis=0)) - 1
            wrist.accel_vm[wrist.accel_vm < 0] = 0
            del wrist.signals

            wrist.timestamps = pd.date_range(start=wrist.header["startdate"],
                                             freq="{}ms".format(round(1000 / fs, 6)), periods=len(wrist.x))

        if not self.wrist_exists:
            wrist = None
            fs = 1

        # Scales Powell et al. 2017 cutpoints to correct epoch length and sample rate (SVMs)
        if self.cutpoints.capitalize() == "Powell":
            print("\nSetting cutpoints to cutpoints from Powell et al. 2017.")
            cutpoint_dict = {"Light": 47 * fs / 30 * self.epoch_len / 15,
                             "Moderate": 64 * fs / 30 * self.epoch_len / 15,
                             "Vigorous": 157 * fs / 30 * self.epoch_len / 15}

        # Cutpoints use AVM and are independent of sample rates and  epoch lengths
        if self.cutpoints.capitalize() == "Fraysse":
            print("\nSetting cutpoints to cutpoints from Fraysse et al. 2021.")
            cutpoint_dict = {"Light": 42.5, "Moderate": 62.5, "Vigorous": 100000}

        return wrist, cutpoint_dict

    def epoch_accel(self, acc_type, fs, vm_data, epoch_len):
        """Epochs accelerometer data. Calculates sum of vector magnitudes (SVM) and average vector magnitude (AVM)
           values for specified epoch length.

           :returns
           -svm: list
           -avm: list
        """

        # Epochs data if read in raw and didn't read in processed data -----------------------------------------------
        if not self.load_raw or self.from_processed:
            return None, None

        if self.load_raw and not self.from_processed:

            try:
                print("\nEpoching {} data into {}-second epochs...".format(acc_type, epoch_len))
                t0 = datetime.datetime.now()

                vm = [i for i in vm_data]
                svm = []
                avm = []

                for i in range(0, len(vm), int(fs * epoch_len)):

                    if i + epoch_len * fs > len(vm):
                        break

                    vm_sum = sum(vm[i:i + epoch_len * fs])
                    avg = vm_sum * 1000 / len(vm[i:i + epoch_len * fs])

                    svm.append(round(vm_sum, 2))
                    avm.append(round(avg, 2))

                t1 = datetime.datetime.now()
                print("Complete ({} seconds)".format(round((t1 - t0).total_seconds(), 1)))

            except TypeError:
                print("\nNo {} file given.".format(acc_type))
                svm, avm = [], []

            return svm, avm

    def calculate_wrist_intensity(self):
        """Calculates activity intensity using wrist cutpoints from Powell et al. (2017). Requires 15-second epochs.
           Calculates total and daily activity volumes."""

        print("\nCalculating activity intensity data using wrist accelerometer...")

        if self.cutpoints.capitalize() == "Powell":
            data = self.df_epoch["WristSVM"]
        if self.cutpoints.capitalize() == "Fraysse":
            data = self.df_epoch["WristAVM"]

        intensity = []
        for i in data:
            if i < self.cutpoint_dict["Light"]:
                intensity.append("Sedentary")
            if self.cutpoint_dict["Light"] <= i < self.cutpoint_dict["Moderate"]:
                intensity.append("Light")
            if self.cutpoint_dict["Moderate"] <= i < self.cutpoint_dict["Vigorous"]:
                intensity.append("Moderate")
            if self.cutpoint_dict["Vigorous"] <= i:
                intensity.append("Vigorous")

        self.df_epoch["WristIntensity"] = intensity

        epoch_to_mins = 60 / self.epoch_len
        values = self.df_epoch["WristIntensity"].value_counts()

        if "Light" not in values.keys():
            values["Light"] = 0
        if "Moderate" not in values.keys():
            values["Moderate"] = 0
        if "Vigorous" not in values.keys():
            values["Vigorous"] = 0

        # TOTAL ACTIVITY ---------------------------------------------------------------------------------------------
        self.activity_totals = {"Sedentary": values["Sedentary"] / epoch_to_mins,
                                "Light": values["Light"] / epoch_to_mins,
                                "Moderate": values["Moderate"] / epoch_to_mins,
                                "Vigorous": values["Vigorous"] / epoch_to_mins,
                                "MVPA": values["Moderate"] / epoch_to_mins + values["Vigorous"] / epoch_to_mins}

        # DAILY ACTIVITY ---------------------------------------------------------------------------------------------
        dates = set([i.date() for i in self.df_epoch["Timestamp"]])
        self.df_epoch["Date"] = [i.date() for i in self.df_epoch["Timestamp"]]

        daily_totals = []

        for date in sorted(dates):
            df = self.df_epoch.loc[self.df_epoch["Date"] == date]
            values = df["WristIntensity"].value_counts()

            if "Light" not in values.keys():
                values["Light"] = 0
            if "Moderate" not in values.keys():
                values["Moderate"] = 0
            if "Vigorous" not in values.keys():
                values["Vigorous"] = 0

            values = values / epoch_to_mins
            daily_data = [date,
                          values["Sedentary"], values["Light"], values["Moderate"],
                          values["Vigorous"], values["Moderate"] + values["Vigorous"]]
            daily_totals.append(daily_data)

        self.df_daily = pd.DataFrame(daily_totals,
                                     columns=["date", "sedentary", "light", "moderate", "vigorous", "mvpa"])

        # Adds totals as final row
        final_row = pd.DataFrame(list(zip(["TOTAL", self.activity_totals["Sedentary"], self.activity_totals["Light"],
                                           self.activity_totals["Moderate"], self.activity_totals["Vigorous"],
                                           self.activity_totals["MVPA"]])),
                                 index=["date", "sedentary", "light", "moderate", "vigorous", "mvpa"]).transpose()
        self.df_daily = self.df_daily.append(final_row)
        self.df_daily = self.df_daily.reset_index()
        self.df_daily = self.df_daily.drop("index", axis=1)

        # Removes date column
        self.df_epoch = self.df_epoch.drop("Date", axis=1)

        # Writing activity totals data --------------------------------------------------------------------------------
        if self.write_intensity_data:
            write_file = False

            file_list = os.listdir(self.output_dir)
            f_name = "{}_Epoch{}{}_DailyActivityVolume.csv".format(self.subj_id, self.epoch_len, self.cutpoints)

            # What to do if file already exists
            if f_name in file_list:

                # If overwrite set to True
                if self.overwrite_output:
                    write_file = True
                    print("Automatically overwritting existing file.")

                # If overwrite set to False, prompts user
                if not self.overwrite_output:
                    user_input = input("Overwrite existing file? y/n: ")

                    if user_input.capitalize() == "Y" or user_input.capitalize() == "Yes":
                        write_file = True

                    if user_input.capitalize() == "N" or user_input.capitalize() == "No":
                        print("File will not be overwritten.")

            # What to do if file does not exist
            if f_name not in file_list:
                write_file = True

            # Writing file?
            if write_file:
                print("Writing total activity volume data to "
                      "{}{}_Epoch{}{}_DailyActivityVolume.csv".format(self.output_dir, self.subj_id,
                                                                      self.epoch_len, self.cutpoints))

                df = self.df_daily.copy()

                df.insert(loc=0, column="ID", value=[self.subj_id for i in range(self.df_daily.shape[0])])

                # Sets wear location based on presence of "LW"/"RW" in filename
                if "LW" in self.wrist_filepath.split("/")[-1]:
                    wear_loc = "LW"
                elif "RW" in self.wrist_filepath.split("/")[-1]:
                    wear_loc = "RW"
                else:
                    wear_loc = "Unknown"

                df.insert(loc=1, column="location", value=[wear_loc for i in range(self.df_daily.shape[0])])
                df.insert(loc=2, column="epoch_len", value=[self.epoch_len for i in range(self.df_daily.shape[0])])
                df.insert(loc=3, column="cutpoints", value=[self.cutpoints for i in range(self.df_daily.shape[0])])

            df.to_csv("{}{}_Epoch{}{}_DailyActivityVolume.csv".format(self.output_dir, self.subj_id,
                                                                      self.epoch_len, self.cutpoints),
                      index=False, float_format='%.2f')

    def create_epoch_df(self, epoch_len=1):
        """Creates dataframe for epoched wrist and ankle data.
           Deletes corresponding data objects for memory management.
           Option to write to .csv and to automatically overwrite existing file. If file is not to be overwritten,
           user is prompted to manually overwrite.

        :argument
        -write_df: boolean

        :returns
        -epoched dataframe: df
        """

        print("\nCombining data into single dataframe...")

        # Sets timestamps if available from wrist; fills in empty data if no Ankle file
        if self.wrist_exists:
            timestamps = pd.date_range(start=self.wrist.timestamps[0], end=self.wrist.timestamps[-1],
                                       freq="{}S".format(epoch_len))

            wrist_svm = [sum(self.wrist.accel_vm[i:int(i+self.wrist.fs*epoch_len)]) for
                         i in range(0, len(self.wrist.accel_vm), int(self.wrist.fs*epoch_len))]

            wrist_avm = [1000*np.mean(self.wrist.accel_vm[i:int(i+self.wrist.fs*epoch_len)]) for
                         i in range(0, len(self.wrist.accel_vm), int(self.wrist.fs*epoch_len))]

            if not self.ankle_exists:
                ankle_svm = [None for i in range(len(timestamps))]
                ankle_avm = [None for i in range(len(timestamps))]

        # Sets timestamps if available from Ankle; fills in empty data if no Wrist file
        if self.ankle_exists and not self.wrist_exists:
            timestamps = pd.date_range(start=self.ankle.timestamps[0], end=self.ankle.timestamps[-1],
                                       freq="{}S".format(epoch_len))
            wrist_svm = [None for i in range(len(timestamps))]
            wrist_avm = [None for i in range(len(timestamps))]

        if self.ankle_exists:
            ankle_svm = [sum(self.ankle.accel_vm[i:int(i+self.ankle.fs*epoch_len)]) for
                         i in range(0, len(self.ankle.accel_vm), int(self.ankle.fs*epoch_len))]

            ankle_avm = [1000*np.mean(self.ankle.accel_vm[i:int(i+self.ankle.fs*epoch_len)]) for
                         i in range(0, len(self.ankle.accel_vm), int(self.ankle.fs*epoch_len))]

        df = pd.DataFrame(list(zip(timestamps, wrist_svm, wrist_avm, ankle_svm, ankle_avm)),
                          columns=["Timestamp", "WristSVM", "WristAVM", "AnkleSVM", "AnkleAVM"])

        return df

    def import_processed_df(self, epoch_len):
        """Imports existing processed 1s epoch data file (.csv) and re-epochs into self.epoch_len epochs.

        :returns
        -dataframe: df
        """

        print("\nImporting existing data ({})".format(self.processed_filepath.split("/")[-1]))

        df = pd.read_csv(self.processed_filepath)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        wrist_svm = [sum(df["WristSVM"].iloc[i:i+epoch_len]) for i in range(0, df.shape[0], epoch_len)]
        wrist_avm = [1000*sum(df["WristSVM"].iloc[i:i+epoch_len]/epoch_len) for i in range(0, df.shape[0], epoch_len)]

        ankle_svm = [sum(df["AnkleSVM"].iloc[i:i+epoch_len]) for i in range(0, df.shape[0], epoch_len)]
        ankle_avm = [1000*sum(df["AnkleSVM"].iloc[i:i+epoch_len]/epoch_len) for i in range(0, df.shape[0], epoch_len)]

        df_long = pd.DataFrame(list(zip(df["Timestamp"].iloc[::epoch_len],
                                        wrist_svm, wrist_avm, ankle_svm, ankle_avm)))
        df_long.columns = ["Timestamp", "WristSVM", "WristAVM", "AnkleSVM", "AnkleAVM"]

        df_long["WristIntensity"] = [None for i in range(df_long.shape[0])]

        return df, df_long

    def write_epoched_df(self):

        write_file = False

        file_list = os.listdir(self.output_dir)
        f_name = "{}_Epoch{}_Accelerometer.csv".format(self.subj_id, self.epoch_len)

        # What to do if file already exists -----------------------------------------------------------------------
        if f_name in file_list:

            # If overwrite set to True
            if self.overwrite_output:
                write_file = True
                print("Automatically overwritting existing file.")

            # If overwrite set to False, prompts user
            if not self.overwrite_output:
                user_input = input("Overwrite existing file? y/n: ")

                if user_input.capitalize() == "Y" or user_input.capitalize() == "Yes":
                    write_file = True

                if user_input.capitalize() == "N" or user_input.capitalize() == "No":
                    print("File will not be overwritten.")

        # What to do if file does not exist -----------------------------------------------------------------------
        if f_name not in file_list:
            write_file = True

        # Writing file? -------------------------------------------------------------------------------------------
        if write_file:
            print("\nWriting epoched data to {}{}_Epoch{}_Accelerometer.csv".format(self.output_dir,
                                                                                    self.subj_id,
                                                                                    self.epoch_len))
            self.df_epoch.to_csv("{}{}_Epoch{}_Accelerometer.csv".format(self.output_dir,
                                                                         self.subj_id,
                                                                         self.epoch_len),
                                 index=False, float_format='%.2f')

            print("\nWriting epoched data to {}{}_Epoch1_Accelerometer.csv".format(self.output_dir,
                                                                                    self.subj_id))
            self.df_epoch1s.to_csv("{}{}_Epoch1_Accelerometer.csv".format(self.output_dir,
                                                                          self.subj_id),
                                   index=False, float_format='%.2f')

    def filter_epoched_data(self, col_name=None, fs=1, filter_type="lowpass", low_f=0.05, high_f=10):

        if filter_type != "bandpass":
            print("\nFiltering {} with {}Hz {} filter...".format(col_name, low_f, filter_type))
        if filter_type == "bandpass":
            print("\nFiltering {} with {}-{}Hz {} filter...".format(col_name, low_f, high_f, filter_type))

        filtered = Filtering.filter_signal(data=self.df_epoch[col_name], filter_type=filter_type,
                                           sample_f=fs, low_f=low_f, high_f=high_f)

        filtered = [i if i >= 0 else 0 for i in filtered]

        self.df_epoch[col_name + "_Filt"] = filtered

        print("Complete.")

    def plot_filtered(self, col_name="AnkleAVM"):

        fig, ax = plt.subplots(1, figsize=(10, 6))
        plt.subplots_adjust(bottom=.125)
        ax.plot(self.df_epoch["Timestamp"], self.df_epoch[col_name],
                color='black', label="Epoch_{}s".format(self.epoch_len))
        ax.plot(self.df_epoch["Timestamp"], self.df_epoch[col_name + "_Filt"], color='red', label="Filtered")
        ax.set_title(col_name)
        ax.legend()

        ax.fill_between(self.df_epoch["Timestamp"], 0, self.df_epoch[col_name + "_Filt"], color='red', alpha=.25)

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)


s = Subject(
            subj_id="OND07_WTL_3034",
            ankle_filepath="/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/Data/Converted/Collection 3/HIIT_GENEActiv_Accelerometer_003_A_LA.edf",
            wrist_filepath="/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/Data/Converted/Collection 3/HIIT_GENEActiv_Accelerometer_003_A_LW.edf",
            load_raw=True,
            epoch_len=15,
            cutpoints="Powell",

            processed_filepath="/Users/kyleweber/Desktop/{}_Epoch1_Accelerometer.csv",
            from_processed=False,

            output_dir="/Users/kyleweber/Desktop/",
            write_epoched_data=False, write_intensity_data=False,
            overwrite_output=True)
