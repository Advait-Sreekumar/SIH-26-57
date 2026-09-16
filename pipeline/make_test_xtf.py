import numpy as np
import pyxtf

OUT = r"D:\SIH\external_test\synthetic_track.xtf"
N_PINGS = 128
N_SAMPLES = 512
RANGE_M = 50.0
ALT_M = 15.0


def make_channel(with_target):
    img = (np.random.rand(N_PINGS, N_SAMPLES) * 80).astype(np.float32)
    for i in range(N_PINGS):
        img[i] *= 0.8 + 0.4 * np.sin(i / 7.0)
    if with_target:
        from scipy.ndimage import gaussian_filter

        hull = np.zeros((N_PINGS, N_SAMPLES), np.float32)
        cv_ellipse = np.zeros((N_PINGS, N_SAMPLES), np.uint8)
        cv_ellipse = cv2_ellipse(hull.shape, center=(365, 65), axes=(38, 16), angle=90)
        hull[cv_ellipse == 1] = 1.0
        hull = gaussian_filter(hull, 3.0)
        texture = np.random.uniform(0.5, 1.5, hull.shape).astype(np.float32)
        texture = gaussian_filter(texture, 1.5)
        target = hull * texture * 900.0
        ridge = np.zeros(hull.shape, np.float32)
        ridge[:, 355:375] = 0.6
        ridge = gaussian_filter(ridge, 2.0)
        target += hull * ridge * 500.0
        img += target
        shadow = np.zeros((N_PINGS, N_SAMPLES), np.float32)
        cv_ellipse_s = cv2_ellipse(shadow.shape, center=(415, 65), axes=(30, 13), angle=90)
        shadow[cv_ellipse_s == 1] = 1.0
        shadow = gaussian_filter(shadow, 4.0)
        img *= 1.0 - 0.9 * shadow
    return np.clip(img, 0, None)


def cv2_ellipse(shape, center, axes, angle):
    import cv2

    canvas = np.zeros(shape, np.uint8)
    cv2.ellipse(canvas, center, axes, angle, 0, 360, 1, -1)
    return canvas


def main():
    file_hdr = pyxtf.XTFFileHeader()
    file_hdr.FileFormat = 0x1234
    file_hdr.SonarName = b"SYNTH_SSS"
    file_hdr.RecordingProgramName = b"shpdemo1"
    file_hdr.NumberOfSonarChannels = 2
    file_hdr.NavUnits = 0
    for ci, name in zip(file_hdr.ChanInfo, (b"PORT", b"STBD")):
        ci.BytesPerSample = 2
        ci.TypeOfChannel = 1
        ci.ChannelName = name
        ci.VoltScale = 1.0

    port = make_channel(False)
    stbd = make_channel(True)

    lat0, lon0 = 45.0000, -83.5000
    with open(OUT, "wb") as f:
        f.write(bytes(file_hdr))
        for i in range(N_PINGS):
            ping_hdr = pyxtf.XTFPingHeader()
            ping_hdr.HeaderType = 0
            ping_hdr.PingNumber = i + 1
            ping_hdr.NumChansToFollow = 2
            ping_hdr.SensorYcoordinate = lat0 + i * 1e-5
            ping_hdr.SensorXcoordinate = lon0
            ping_hdr.SensorHeading = 0.0
            ping_hdr.SensorPrimaryAltitude = ALT_M
            ping_hdr.RangeToFish = int(RANGE_M)
            ping_hdr.SoundVelocity = 1500.0
            ping_hdr.SensorSpeed = 1.5

            chans = []
            for cn, data in ((0, port[i]), (1, stbd[i])):
                chan_hdr = pyxtf.XTFPingChanHeader()
                chan_hdr.ChannelNumber = cn
                chan_hdr.NumSamples = N_SAMPLES
                chan_hdr.SlantRange = RANGE_M
                chan_hdr.TimeDelay = 0.0
                chans.append((chan_hdr, data.astype(np.uint16)))

            ping_hdr.NumBytesThisRecord = 256 + sum(64 + d.nbytes for _, d in chans)
            f.write(bytes(ping_hdr))
            for chan_hdr, data in chans:
                f.write(bytes(chan_hdr))
                f.write(data.tobytes())
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
