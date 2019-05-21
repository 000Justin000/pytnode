import json
import dateutil.parser
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

if __name__ == "__main__":

    with open("events.json") as f: 
        dat = json.load(f)

    events = dat[0]
    reviewed = dat[1]

    #------------------------------------------
    # human label
    #------------------------------------------
    i = 0
    input2sentiment = {'4': 1.0, '1': 0.5, '3': 0.0, '2': -0.5, '5': -1.0}
    #------------------------------------------
    for i in range(len(events)):
        if not reviewed[i]:
            print(events[i][-1])

            correct_input = False
            while (not correct_input):
                input_value = input()
                if input_value in input2sentiment:
                    sentiment = input2sentiment[input_value]
                    correct_input = True
                else:
                    print("retry --- 4: 1.0, 1: 0.5, 3: 0.0, 2: -0.5, 5: -1.0")

            events[i] = events[i][0:1] + ["{:+10.4f}".format(sentiment)] + events[i][2:]
            reviewed[i] = True

            if i % 5 == 0:
                json.dump([events, reviewed], open("events.json", "w"))
                np.savetxt("events_manual", events, fmt=["%s", "%s", "%s"], delimiter='\t')
    #------------------------------------------
