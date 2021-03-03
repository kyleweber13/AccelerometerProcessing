import pyedflib
import datetime
from datetime import timedelta
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import AccelProcessing.Accelerometer as Accelerometer


class Subject:

    def __init__(self, subj_id=None, wrist_filepath=None, ankle_filepath=None, processed_filepath=None,
                 load_raw=True, from_processed=False, output_dir=None, epoch_len=15, cutpoints="Powell",
                 write_epoched_data=False, write_intensity_data=False, overwrite_output=False):
        """Class to read in EDF-formatted wrist and ankle accelerometer files.

        :argument
        -subj_id: used for file naming, str
            -format example: "OND07_WTL_3013"
        -wrist_filepath: full pathway to wrist .edf file
        -ankle_filepath: full pathway to ankle .edf file
        -processed_filepath: full pathway to .csv file created using Subject.create_epoch_df()
        -load_raw: whether to load raw data; boolean
        -from_processed: whether to load file specified using processed_filepath; boolean
        -output_dir: full pathway to where files get written
        -epoch_len: epoch length in seconds, int
        -cutpoints: str for which cutpoints to use
            -"Powell" for Powell et al. 2017 or "Fraysse" for Fraysse et al. 2021
        -write_epoched_data: whether to write df_epoch to .csv; boolean
        -write_intensity_data: whether to write df_daily and activity_totals to .csv's; boolean
        -overwrite_output: whether to automatically overwrite existing df_epoch file; boolean
            -If False, user will be prompted to manually overwrite existing file.
        """

        self.subj_id = subj_id

        if wrist_filepath is not None:
            self.wrist_filepath = wrist_filepath.format(subj_id.split("_")[-1])
        if wrist_filepath is None:
            self.wrist_filepath = None

        if ankle_filepath is not None:
            self.ankle_filepath = ankle_filepath.format(subj_id.split("_")[-1])
        if ankle_filepath is None:
            self.ankle_filepath = None

        self.processed_filepath = processed_filepath.format(subj_id.split("_")[-1])
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

            if self.wrist_filepath is not None:
                self.wrist, self.cutpoint_dict = self.create_wrist_obj()

                self.wrist_svm, self.wrist_avm = self.epoch_accel(acc_type="wrist",
                                                                  fs=self.wrist.sample_rate,
                                                                  vm_data=self.wrist.accel_vm)

            if self.wrist_filepath is None:
                self.wrist = None
                self.cutpoint_dict = None
                self.wrist_svm = None
                self.wrist_avm = None

            self.ankle = self.create_ankle_obj()

            self.ankle_svm, self.ankle_avm = self.epoch_accel(acc_type="ankle",
                                                              fs=self.ankle.sample_rate,
                                                              vm_data=self.ankle.accel_vm)

            self.df_epoch = self.create_epoch_df()

        if self.from_processed:
            self.df_epoch = self.import_processed_df()

        if self.wrist_filepath is not None:
            self.calculate_wrist_intensity()

        if self.write_epoched:
            self.write_epoched_df()

        print("\n====================================================================================================")
        print("Processing complete.")

    def print_summary(self):
        """Prints summary of what data will be read in."""

        print("======================================================================================================")
        print("\nData import summary:")

        if self.wrist_filepath is not None:
            print("-Importing wrist file: {}".format(self.wrist_filepath))
        if self.wrist_filepath is None:
            print("-No wrist file will be imported.")

        if self.ankle_filepath is not None:
            print("-Importing ankle file: {}".format(self.ankle_filepath))
        if self.ankle_filepath is None:
            print("-No ankle file will be imported.")

        if self.load_raw:
            print("\n-Raw data will be imported.")
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

    def create_wrist_obj(self):
        """Creates wrist accelerometer data object.
           Scales accelerometer cutpoints from Powell et al. (2017) to selected epoch length.

        :returns
        -wrist object
        -cutpoints: dictionary
        """

        print("\n--------------------------------------------- Wrist file --------------------------------------------")
        if self.wrist_filepath is not None:
            wrist = Accelerometer.Accelerometer(raw_filepath=self.wrist_filepath,
                                                load_raw=self.load_raw,
                                                start_offset=self.wrist_offset)
            fs = wrist.sample_rate

        if self.wrist_filepath is None:
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
            cutpoint_dict = {"Light": 42.5, "Moderate": 62.5, "Vigorous": 10000}

        return wrist, cutpoint_dict

    def create_ankle_obj(self):
        """Creates ankle accelerometer data object.

        :returns
        -ankle object
        """

        print("\n--------------------------------------------- Ankle file --------------------------------------------")

        ankle = Accelerometer.Accelerometer(raw_filepath=self.ankle_filepath,
                                            load_raw=self.load_raw,
                                            start_offset=self.ankle_offset)

        return ankle

    def epoch_accel(self, acc_type, fs, vm_data):
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

            print("\nEpoching {} data into {}-second epochs...".format(acc_type, self.epoch_len))
            t0 = datetime.datetime.now()

            vm = [i for i in vm_data]
            svm = []
            avm = []

            for i in range(0, len(vm), int(fs * self.epoch_len)):

                if i + self.epoch_len * fs > len(vm):
                    break

                vm_sum = sum(vm[i:i + self.epoch_len * fs])
                avg = vm_sum * 1000 / len(vm[i:i + self.epoch_len * fs])

                svm.append(round(vm_sum, 2))
                avm.append(round(avg, 2))

            t1 = datetime.datetime.now()
            print("Complete ({} seconds)".format(round((t1 - t0).total_seconds(), 1)))

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
            f_name = "{}_DailyActivityVolume.csv".format(self.subj_id)

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
                      "{}{}_DailyActivityVolume.csv".format(self.output_dir, self.subj_id))

                df = self.df_daily.copy()

                df.insert(loc=0, column="ID", value=[self.subj_id for i in range(self.df_daily.shape[0])])
                wear_loc = self.wrist_filepath.split("/")[-1].split(".")[0].split("_")[-2]
                df.insert(loc=1, column="location", value=[wear_loc for i in range(self.df_daily.shape[0])])
                df.insert(loc=2, column="epoch_len", value=[self.epoch_len for i in range(self.df_daily.shape[0])])
                df.insert(loc=3, column="cutpoints", value=[self.cutpoints for i in range(self.df_daily.shape[0])])

            df.to_csv("{}{}_DailyActivityVolume.csv".format(self.output_dir, self.subj_id),
                      index=False, float_format='%.2f')

    def create_epoch_df(self):
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

        if self.wrist_filepath is not None:
            timestamps = pd.date_range(start=self.wrist.timestamps[0], end=self.wrist.timestamps[-1],
                                       freq="{}S".format(self.epoch_len))

        if self.ankle_filepath is not None and self.wrist_filepath is None:
            timestamps = pd.date_range(start=self.ankle.timestamps[0], end=self.ankle.timestamps[-1],
                                       freq="{}S".format(self.epoch_len))
            self.wrist_svm = [None for i in range(len(timestamps))]
            self.wrist_avm = [None for i in range(len(timestamps))]

        df = pd.DataFrame(list(zip(timestamps, self.wrist_svm, self.wrist_avm, self.ankle_svm, self.ankle_avm)),
                          columns=["Timestamp", "WristSVM", "WristAVM", "AnkleSVM", "AnkleAVM"])

        del self.wrist_svm, self.wrist_avm, self.ankle_svm, self.ankle_avm

        return df

    def import_processed_df(self):
        """Imports existing processed epoch data file (.csv).

        :returns
        -dataframe: df
        """

        print("\nImporting existing data ({})".format(self.processed_filepath.split("/")[-1]))

        df = pd.read_csv(self.processed_filepath)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        return df

    def write_epoched_df(self):

        write_file = False

        file_list = os.listdir(self.output_dir)
        f_name = "{}_EpochedAccelerometer.csv".format(self.subj_id)

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
            print("Writing epoched data to {}{}_EpochedAccelerometer.csv".format(self.output_dir, self.subj_id))
            self.df_epoch.to_csv("{}{}_EpochedAccelerometer.csv".format(self.output_dir, self.subj_id),
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


s = Subject(subj_id="OND07_WTL_3034",
            # ankle_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_GA_LAnkle_Accelerometer.EDF",
            wrist_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_GA_LWrist_Accelerometer.EDF",
            load_raw=True,
            epoch_len=15,
            cutpoints="Fraysse",

            processed_filepath="/Users/kyleweber/Desktop/OND07_ProcessedAnkle/OND07_WTL_{}_EpochedAccelerometer.csv",
            from_processed=False,

            output_dir="/Users/kyleweber/Desktop/",
            write_epoched_data=True, write_intensity_data=True,
            overwrite_output=False)

# Update None read-ins
