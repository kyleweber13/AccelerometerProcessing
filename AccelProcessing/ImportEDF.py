import pyedflib
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class GENEActiv:

    def __init__(self, filepath, load_raw, start_offset=0, end_offset=0):

        self.filepath = filepath
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.load_raw = load_raw

        # Accelerometer data
        self.x = None
        self.y = None
        self.z = None
        self.vm = None  # Vector Magnitudes
        self.timestamps = None

        # Details
        self.sample_rate = 75  # default value
        self.starttime = None
        self.file_dur = None

        # IMPORTS GENEActiv FILE
        if self.load_raw:
            self.import_file()

    def import_file(self):

        t0 = datetime.now()  # Gets current time

        print("Importing {}...".format(self.filepath))

        # READS IN ACCELEROMETER DATA ================================================================================
        file = pyedflib.EdfReader(self.filepath)

        if self.end_offset != 0:
            print("Importing file from index {} to {}...".format(self.start_offset, self.end_offset))

            self.x = file.readSignal(chn=0, start=self.start_offset, n=self.end_offset)
            self.y = file.readSignal(chn=1, start=self.start_offset, n=self.end_offset)
            self.z = file.readSignal(chn=2, start=self.start_offset, n=self.end_offset)

        if self.end_offset == 0:
            self.x = file.readSignal(chn=0, start=self.start_offset)
            self.y = file.readSignal(chn=1, start=self.start_offset)
            self.z = file.readSignal(chn=2, start=self.start_offset)

        # Calculates gravity-subtracted vector magnitude
        # Negative values become zero
        self.vm = np.sqrt(np.square(np.array([self.x, self.y, self.z])).sum(axis=0)) - 1
        self.vm[self.vm < 0] = 0

        self.sample_rate = file.getSampleFrequencies()[1]  # sample rate
        self.starttime = file.getStartdatetime() + timedelta(seconds=self.start_offset/self.sample_rate)
        self.file_dur = round(file.getFileDuration() / 3600, 3)  # Seconds --> hours

        # TIMESTAMP GENERATION ========================================================================================
        t0_stamp = datetime.now()

        print("\n" + "Creating timestamps...")

        end_time = self.starttime + timedelta(seconds=len(self.x) / self.sample_rate)
        self.timestamps = np.asarray(pd.date_range(start=self.starttime, end=end_time, periods=len(self.x)))

        t1_stamp = datetime.now()
        stamp_time = (t1_stamp-t0_stamp).seconds
        print("Complete ({} seconds).".format(round(stamp_time, 2)))

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("Import complete ({} seconds).".format(round(proc_time, 2)))
