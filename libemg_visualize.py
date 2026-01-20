if __name__ == "__main__":
    from libemg.data_handler import OnlineDataHandler
    from libemg.streamers import emager_streamer
    from libemg.filtering import Filter
    import time
    # from utils.find_usb import virtual_port
    from config_emager import *


    # Create data handler and streamer
    p, smi = emager_streamer()
    print(f"Streamer created: process: {p}, smi : {smi}")
    odh = OnlineDataHandler(shared_memory_items=smi)

    if FILTER:
        filter = Filter(SAMPLING)
        notch_filter_dictionary={ "name": "notch", "cutoff": 60, "bandwidth": 3}
        filter.install_filters(notch_filter_dictionary)
        bandpass_filter_dictionary={ "name":"bandpass", "cutoff": [20, 450], "order": 4}
        filter.install_filters(bandpass_filter_dictionary)
        odh.install_filter(filter)

    try :
        odh.visualize(num_samples=5000, block=True)
    except Exception as e:
        print(e)
    finally:
        print("Exiting...")