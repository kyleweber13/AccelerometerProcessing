import AccelProcessing.ImportEDF as ImportEDF


class Accelerometer:

    def __init__(self, raw_filepath=None, load_raw=True, start_offset=0):

        self.filepath = raw_filepath
        self.load_raw = load_raw
        self.sample_rate = 1
        self.starttime = None
        self.start_offset = start_offset

        self.timestamps, self.accel_x, self.accel_y, self.accel_z, \
        self.accel_vm, self.sample_rate, self.starttime = self.load_raw_data()

    def load_raw_data(self):

        if self.filepath is None:
            return None, None, None, None, None, None, None

        accel = ImportEDF.GENEActiv(filepath=self.filepath, load_raw=self.load_raw,
                                    start_offset=self.start_offset, end_offset=0)

        return accel.timestamps, accel.x, accel.y, accel.z, accel.vm, accel.sample_rate, accel.starttime
