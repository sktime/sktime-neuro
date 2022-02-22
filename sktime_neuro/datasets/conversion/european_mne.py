import matplotlib.pyplot as plt
import mne.io
import pooch
from typing import List, Dict
import sktime_neuro.utils.mne_processing
from sktime.datasets._data_io import write_dataframe_to_tsfile
from sktime.datasets._single_problem_loaders import load_UCR_UEA_dataset
import time



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
    #Mne is probably the worst implementation possible for what it does.
    [af,bf] = mne.datasets.sleep_physionet.age.fetch_data(subjects=[0,1], recording=[1], path="edfdatasets/") # Temp path for now
    #Af and bf are both different people's sleep sample, we only test these 2 for now
    #Currently this is very brittle and will break if the datasets are unavaliable
    raw = mne.io.read_raw_edf(af[0], stim_channel="marker", misc=["rectal"])
    annotation = mne.read_annotations(af[1])
    raw.set_annotations(annotation)

    eventList = mne.events_from_annotations(raw)[0] #Times (in samples) and the class they belong to
    labels = mne.events_from_annotations(raw)[1] # Names for the class labels
    #need to find sampel rate, and data for each channel, and write that into a ts file with the annotations (class label)


    #test
    AH = load_UCR_UEA_dataset("ArrowHead")

    channelList = []
    #Get the channel ids, the names and the data and put it in one nice list
    for i in range(len(raw.ch_names)):
        channelList.append({"ID" : i, "Name" : raw.ch_names[i], "Data" : raw[i]})
        print(channelList[i])
    print("Last sample = " + str(raw.last_samp))
    print("Samp freq = " + str(raw.info['sfreq']))
    print(len(channelList[0]['Data'][0][0])) #Gets the total samples, Have to dive a bit to get some of this info, blame nparrays
    print(eventList[1][0]) # Get start of 2nd event
    print(len(eventList)) # Get num events
    events_from_annot, events = mne.events_from_annotations(raw)


    epochs_train = mne.Epochs(raw=raw, events=events_from_annot,event_id=None, baseline=None) # Not useful
    print("d")

    before = time.time()
    classData = []
    counter = 0
    #Iteerate though all channels and store (seperate each channel) until we reach a new event, in which create new array
    while counter < len(eventList): #While we still got events to process
        classData.append([])

        #list subthingy
        min = eventList[counter][0]
        if counter + 1 >= len(eventList):
            max = len(channelList[0]['Data'][0][0]) ## Total samples
        else:
            max = eventList[counter+1][0]
        tempArr = [eventList[counter][2]] # Set what class it belongs to
        for i in range(len(channelList)): #Iterate through the channels
            tempArr.append(channelList[i]['Data'][0][0][min:max])
        classData[-1].append(tempArr) #Append to current working list
        counter += 1

        print(counter)
    total = time.time() - before
    print(total)
    #chunk up the data for each channel into each class, and then store it in a list
    #write_dataframe_to_tsfile()