import json
def parse_trace_json(trace_file, search_str, search_str_or = "nothing couldbeNoneNOnkdfjaklfjdkajl xx123"):
    with open(trace_file, 'r') as file:
        trace_data = json.load(file)
    
    kernel_events = [event for event in trace_data['traceEvents']
                     if 'cat' in event.keys() 
                     and event['cat'] == 'kernel' 
                     and 'name' in event.keys() 
                     and (search_str in event['name']
                          or search_str_or in event['name'])]
    durations = [float(event['dur']) for event in kernel_events]
    
    # print(durations)
    durations.sort()
    durations = durations[2:-2]
    if durations:
        average_dur = sum(durations) / len(durations)
    else:
        average_dur = -1
    return average_dur