import matplotlib.pyplot as plt
import mne.io
import pooch
from typing import List, Dict
import sktime_neuro.utils.mne_processing


def import_european(path):
    mne.io.read_raw_edf(path)


def import_european_annotations(path):
    mne.read_annotations(path)


def get_channel(channelName: str, dataIndex: int, data) -> mne.io.edf.edf.RawEDF:
    return mne.io.read_raw_edf(data[dataIndex], stim_channel=channelName)


def get_channel_with_misc(channelName: str, miscNames: List[str], dataIndex: int, data) -> mne.io.edf.edf.RawEDF:
    return mne.io.read_raw_edf(data[dataIndex], stim_channel=channelName, misc=miscNames)


def get_annotations(data):
    return mne.read_annotations(data[1])


def get_events(raw, wantedEvents: Dict[str, int]):
    events, _ = mne.events_from_annotations(raw, event_id=wantedEvents)
    return events


def get_events_chunked(raw, wantedEvents: Dict[str, int], chunkLen: int):
    events, _ = mne.events_from_annotations(raw, event_id=wantedEvents, chunk_duration=chunkLen)
    return events

if __name__ == "__main__":
    [af,bf] = mne.datasets.sleep_physionet.age.fetch_data(subjects=[0,1], recording=[1])
    raw = mne.io.read_raw_edf(af[0], stim_channel="marker", misc=["rectal"])
    annotation = sktime_neuro.utils.mne_processing.create_annotation(raw)
    print("A")