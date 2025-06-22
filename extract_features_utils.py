import json
import librosa
import numpy as np
from parselmouth.praat import call
from sklearn.cluster import KMeans
from collections import defaultdict

# Helper methods for feature extraction in Praat
# Imported functions from extract_features_utils.py

def clip_audio(audio, onset=0.25, offset=0.75):
    """Trim audio with librosa and return the middle of the audio.
    """
    trimmed_audio, idx  = librosa.effects.trim(audio)
    start = round(len(trimmed_audio) * onset)
    end = round(len(trimmed_audio) * offset)
    
    return trimmed_audio[start:end], idx[start:end]

def get_f0(sound, minF0, maxF0):
    duration = sound.get_total_duration()
    pitch = call(sound, "To Pitch", 0.0, minF0, maxF0) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0.0, duration, "Hertz") # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0.0, duration, "Hertz")
    minF0 = call(pitch, "Get minimum", 0.0, duration, "Hertz", "Parabolic")
    maxF0 = call(pitch, "Get maximum", 0.0, duration, "Hertz", "Parabolic")

    return meanF0, stdevF0, minF0, maxF0

def get_formants(sound, minF0, maxF0, gender=0):
    """Get F1, F2, F3 metrics from sound.
    If gender = 0, use male focused formant parameters, if gender = 1, use female focused formant params."""
    pointProcess = call(sound, "To PointProcess (periodic, cc)", minF0, maxF0)
    
    if gender == 0:
       formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    else:
       formants = call(sound, "To Formant (burg)", 0.0025, 5, 5500, 0.025, 50)

    numPoints = call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
    return f1_list, f2_list, f3_list
    

def get_jitter(sound, minF0, maxF0):
    """Get jitter related metrics from sound."""
    pointProcess = call(sound, "To PointProcess (periodic, cc)", minF0, maxF0)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    # ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = 3 * rapJitter
    
    return localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter
    

def get_shimmer(sound, minF0, maxF0):
    """Get shimmer related metrics from sound."""
    pointProcess = call(sound, "To PointProcess (periodic, cc)", minF0, maxF0)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    # apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    #ddpShimmer = # call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddpShimmer = 3 * apq3Shimmer
    
    return localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, ddpShimmer


def get_spatial_jitter(times):
    diffs = np.array([t2 - t1 for t1, t2 in zip(times, times[1:])])
    periods = np.array([end - start for start, end in zip(times, times[2:])])
    period_diffs = np.array([np.abs(p2 - p1) for p1, p2 in zip(periods, periods[1:])])
    avg_period = np.mean(periods)
    
    neighbours3 = [[np.abs(p0 - np.mean([p_n1, p0, p_t1]))] for p_n1, p0, p_t1 in zip(times, times[1:], times[2:])]
    neighbours5 = [[np.abs(p0 - np.mean([p_n2, p_n1, p0, p_t1, p_t2]))] for p_n2, p_n1, p0, p_t1, p_t2 in zip(times, times[1:], times[2:], times[3:], times[4:])]
    
    local_jitter = np.mean(diffs) / avg_period
    absolute_jitter = np.mean(diffs)
    jitter_rap = np.mean(neighbours3) / avg_period
    jitter_rap5 = np.mean(neighbours5) / avg_period
    jitter_ddp = np.mean(period_diffs) / avg_period
    
    return local_jitter, absolute_jitter, jitter_rap, jitter_rap5, jitter_ddp


def get_interval_indices(timestamps, start_interval=0.25, end_interval=0.75):
    total_time = timestamps[-1] - timestamps[0]
    start_time = start_interval * total_time
    end_time = end_interval * total_time
    median_indices = [i for i, t in enumerate(timestamps) if start_time <= t <= end_time]
    
    return median_indices[0], median_indices[-1]


def get_spatial_jitter2(times, labels):
    time_diffs = np.array([t2 - t1 for t1, t2 in zip(times, times[1:])])
    periods = []
    prev_time = {}
    period_label = []
    
    for time, label in zip(times, labels):
        if label not in prev_time:
            prev_time[label] = None
        
        if prev_time[label] is not None:
            periods.append(time - prev_time[label])
            period_label.append(label)
        
        prev_time[label] = time
    
    periods = np.array(periods)
    period_diffs = np.array([np.abs(p2 - p1) for p1, p2 in zip(periods, periods[1:])])

    avg_period = np.mean(periods)
    neighbours3 = [[np.abs(p0 - np.mean([p_n1, p0, p_t1]))] for p_n1, p0, p_t1 in zip(times, times[1:], times[2:])]
    neighbours5 = [[np.abs(p0 - np.mean([p_n2, p_n1, p0, p_t1, p_t2]))] for p_n2, p_n1, p0, p_t1, p_t2 in zip(times, times[1:], times[2:], times[3:], times[4:])]
    
    local_jitter = np.mean(time_diffs) / avg_period
    absolute_jitter = np.mean(time_diffs)
    jitter_rap = np.mean(neighbours3) / avg_period
    jitter_rap5 = np.mean(neighbours5) / avg_period
    jitter_ddp = np.mean(period_diffs) / avg_period

    return local_jitter, absolute_jitter, jitter_rap, jitter_rap5, jitter_ddp

    
def get_spatial_shimmer(points):

    
    diffs = np.array([np.linalg.norm(t2 - t1) for t1, t2 in zip(points, points[1:])])
    periods = np.array([np.linalg.norm(end - start) for start, end in zip(points, points[2:])])
    period_diffs = np.array([np.abs(p2 - p1) for p1, p2 in zip(periods, periods[1:])])
    avg_period = np.mean(periods)
    
    neighbours3 = [[np.abs(p0 - np.mean([p_n1, p0, p_t1]))] for p_n1, p0, p_t1 in zip(periods, periods[1:], periods[2:])]
    neighbours5 = [[np.abs(p0 - np.mean([p_n2, p_n1, p0, p_t1, p_t2]))] for p_n2, p_n1, p0, p_t1, p_t2 in zip(points, points[1:], points[2:], points[3:], points[4:])]
    
    shimmer_local = np.mean(diffs) / avg_period
    shimmer_absolute = np.mean(diffs)
    shimmer_rap = np.mean(neighbours3) / avg_period
    shimmer_rap5 = np.mean(neighbours5) / avg_period
    shimmer_ddp = np.mean(period_diffs) / avg_period
    
    return shimmer_local, shimmer_absolute, shimmer_rap, shimmer_rap5, shimmer_ddp


def get_spatial_shimmer2(points, labels):
    diffs = np.array([np.linalg.norm(t2 - t1) for t1, t2 in zip(points, points[1:])])

    periods = []
    prev_point = {}
    period_label = []
    
    for point, label in zip(points, labels):
        if label not in prev_point:
            prev_point[label] = None
        
        if prev_point[label] is not None:
            periods.append(point - prev_point[label])
            period_label.append(label)
        
        prev_point[label] = point

    periods = np.array([np.linalg.norm(end - start) for start, end in zip(points, points[2:])])
    period_diffs = np.array([np.abs(p2 - p1) for p1, p2 in zip(periods, periods[1:])])
    avg_period = np.mean(periods)
    
    neighbours3 = [[np.abs(p0 - np.mean([p_n1, p0, p_t1]))] for p_n1, p0, p_t1 in zip(periods, periods[1:], periods[2:])]
    neighbours5 = [[np.abs(p0 - np.mean([p_n2, p_n1, p0, p_t1, p_t2]))] for p_n2, p_n1, p0, p_t1, p_t2 in zip(points, points[1:], points[2:], points[3:], points[4:])]
    
    shimmer_local = np.mean(diffs) / avg_period
    shimmer_absolute = np.mean(diffs)
    shimmer_rap = np.mean(neighbours3) / avg_period
    shimmer_rap5 = np.mean(neighbours5) / avg_period
    shimmer_ddp = np.mean(period_diffs) / avg_period
    
    return shimmer_local, shimmer_absolute, shimmer_rap, shimmer_rap5, shimmer_ddp


def get_harmonic_to_noise_ratio(sound, minF0):
    """Get harmonic-to-noise ratio from sound."""
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, minF0, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    return hnr


def parse_string_to_vector(s):
    """Parse TapCoordinat into numpy 2-D vectors."""
    s = s.strip("{}")
    parts = s.split(",")
    x, y = float(parts[0]), float(parts[1])
    return np.array([x, y])


def get_point_clusters(points):
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    kmeans.fit(points)
    return kmeans.labels_


def read_tap_json(filename):
    with open(filename) as f:
        data = json.load(f)
    
    timestamps = []
    points = []
    missed_taps = 0
    for tap in data:
        # Get time stamps
        timestamps.append(tap["TapTimeStamp"])
        
        # Get points
        points.append(parse_string_to_vector(tap["TapCoordinate"]))
        
        # Get accuracy
        if tap["TappedButtonId"] == "TappedButtonNone":
            missed_taps += 1
            
    tap_accuracy = (missed_taps - len(data)) / len(data)
        
    return timestamps, points, tap_accuracy

def read_acceleration_json(filename):
    with open(filename) as f:
        data = json.load(f)
    
    timestamps = []
    points = []
    
    for dp in data:
        # Get timestamps
        timestamps.append(dp["timestamp"])
        
        # Get acceleration
        x, y, z = dp["x"], dp["y"], dp["z"]
        point = np.array([x, y, z])
        points.append(point)
        
    return timestamps, points
        
def read_motion_json(filename):
    with open(filename) as f:
        data = json.load(f)
    
    timestamps = []
    attitudes = defaultdict(list)
    accelerations = defaultdict(list)
    rotations = defaultdict(list)
    
    for dp in data:
        # Get timestamps
        timestamps.append(dp["timestamp"])
        
        # Get attitudes
        att = dp["attitude"]
        att_x, att_y, att_z, att_w = att["x"], att["y"], att["z"], att["w"]
        attitudes["x"].append(att_x)
        attitudes["y"].append(att_y)
        attitudes["z"].append(att_z)
        attitudes["w"].append(att_w)
        
        # Get accelerations
        accel = dp["userAcceleration"]
        accel_x, accel_y, accel_z = accel["x"], accel["y"], accel["z"]
        accelerations["x"].append(accel_x)
        accelerations["y"].append(accel_y)
        accelerations["z"].append(accel_z)
        
        # Get rotations
        rott = dp["rotationRate"]
        rott_x, rott_y, rott_z = rott["x"], rott["y"], rott["z"]
        rotations["x"].append(rott_x)
        rotations["y"].append(rott_y)
        rotations["z"].append(rott_z)
        
    return timestamps, attitudes, accelerations, rotations

            
# if __name__ == '__main__':
    
#    filename = librosa.ex('trumpet')
#    y, sr = librosa.load(filename, sr=11025)