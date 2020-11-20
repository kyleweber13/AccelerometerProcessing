import pyedflib
import datetime
from datetime import timedelta
from Accelerometer import Accelerometer
import matplotlib.pyplot as plt
import pandas as pd
import os


class Subject:

    def __init__(self, subj_id=None, wrist_filepath=None, ankle_filepath=None, processed_filepath=None,
                 load_raw=True, from_processed=False, output_dir=None, epoch_len=15,
                 write_epoched_data=False, overwrite_output=False):

        self.subj_id = subj_id
        self.wrist_filepath = wrist_filepath.format(subj_id.split("_")[-1])
        self.ankle_filepath = ankle_filepath.format(subj_id.split("_")[-1])
        self.processed_filepath = processed_filepath
        self.output_dir = output_dir
        self.load_raw = load_raw
        self.from_processed = from_processed
        self.epoch_len = epoch_len
        self.write_epoched = write_epoched_data
        self.overwrite_output = overwrite_output

        # ================================================== RUNS METHODS =============================================
        self.print_summary()

        print("\nImporting data file(s)...\n")

        self.wrist_offset, self.ankle_offset = self.sync_starts()

        self.wrist, self.cutpoint_dict = self.create_wrist_obj()

        self.wrist_svm, self.wrist_avm = self.epoch_accel(acc_type="wrist",
                                                          fs=self.wrist.sample_rate,
                                                          vm_data=self.wrist.accel_vm)

        self.ankle = self.create_ankle_obj()

        self.ankle_svm, self.ankle_avm = self.epoch_accel(acc_type="ankle",
                                                          fs=self.ankle.sample_rate,
                                                          vm_data=self.ankle.accel_vm)

        self.df_epoch = self.create_epoch_df(write_df=self.write_epoched)

        print("\n====================================================================================================")
        print("Processing complete.")

    def print_summary(self):

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
        """Calculates file duration with start and end times. Prints results to console."""

        if filepath is None:
            return None, None, None

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

        return start_time, end_time, edf_file.getSampleFrequency(0)
    
    def sync_starts(self):

        a_start, a_end, a_sample = self.check_file(filepath=self.ankle_filepath, print_summary=False)
        w_start, w_end, w_sample = self.check_file(filepath=self.wrist_filepath, print_summary=False)

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

        print("\n--------------------------------------------- Wrist file --------------------------------------------")
        wrist = Accelerometer(raw_filepath=self.wrist_filepath,
                              processed_filepath=None,
                              output_dir=self.output_dir,
                              epoch_len=self.epoch_len,
                              load_raw=self.load_raw,
                              start_offset=self.wrist_offset,
                              from_processed=self.from_processed,
                              overwrite_without_asking=False)

        cutpoint_dict = {"Light": 47 * wrist.sample_rate * self.epoch_len / 15,
                         "Moderate": 64 * wrist.sample_rate * self.epoch_len / 15,
                         "Vigorous": 157 * wrist.sample_rate * self.epoch_len / 15}

        return wrist, cutpoint_dict

    def create_ankle_obj(self):

        print("\n--------------------------------------------- Ankle file --------------------------------------------")

        ankle = Accelerometer(raw_filepath=self.ankle_filepath,
                              processed_filepath=None,
                              output_dir=self.output_dir,
                              epoch_len=self.epoch_len,
                              load_raw=self.load_raw,
                              from_processed=self.from_processed,
                              start_offset=self.ankle_offset,
                              overwrite_without_asking=False)

        return ankle

    def epoch_accel(self, acc_type, fs, vm_data):

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

    def create_epoch_df(self, write_df=False):

        print("\nCombining data into single dataframe...")

        if self.wrist_filepath is not None:
            timestamps = self.wrist.timestamps[::self.epoch_len * self.wrist.sample_rate]

        if self.ankle_filepath is not None and self.wrist_filepath is None:
            timestamps = self.ankle.timestamps[::self.epoch_len * self.ankle.sample_rate]

        df = pd.DataFrame(list(zip(timestamps, self.wrist_svm, self.wrist_avm, self.ankle_svm, self.ankle_avm)),
                          columns=["Timestamp", "WristSVM", "WristAVM", "AnkleSVM", "AnkleAVM"])

        del self.wrist_svm, self.wrist_avm, self.ankle_svm, self.ankle_avm

        if write_df:
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
                df.to_csv("{}{}_EpochedAccelerometer.csv".format(self.output_dir, self.subj_id),
                          index=False, float_format='%.2f')

        return df


s = Subject(subj_id="OND07_WTL_3013",
            ankle_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_GA_LAnkle_Accelerometer.EDF",
            wrist_filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_GA_LWrist_Accelerometer.EDF",
            load_raw=True,
            epoch_len=15,
            from_processed=False,
            processed_filepath=None,
            output_dir="/Users/kyleweber/Desktop/",
            write_epoched_data=True, overwrite_output=True)
