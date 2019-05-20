import json
import dateutil.parser
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

if __name__ == "__main__":

    with open("eventnews.json") as f: 
        dat = json.load(f)

    events = dat[0]
    reviewed = dat[1]

    #------------------------------------------
    # human label
    #------------------------------------------
    i = 0
    input2sentiment = {'a':0.0, 's':1.0, 'd':-1.0}
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
                    print("retry --- a: 0.0, s: 1.0, d: -1.0")

            events[i] = events[i][0:1] + ["{:+10.4f}".format(sentiment)] + events[i][2:]
            reviewed[i] = True

            if i % 10 == 0:
                json.dump([events, reviewed], open("eventnews.json", "w"))
                np.savetxt("eventmanual", events, fmt=["%s", "%s", "%s"], delimiter='\t')
    #------------------------------------------
