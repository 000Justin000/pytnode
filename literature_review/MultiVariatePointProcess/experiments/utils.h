void ImportFromFileExistingSequences(const std::string& filename, std::vector<Sequence>& data)
{
    std::ifstream fin(filename.c_str());
    std::string str;
    unsigned seqID = 0;

    std::cout << filename << std::endl;
    while(std::getline(fin, str))
    {
        std::istringstream istr(str.substr(0, str.size()-1));
        std::vector<std::string> parts{std::istream_iterator<std::string>{istr}, 
                                       std::istream_iterator<std::string>{}};

        Sequence seq;
        unsigned eventID = 0;

        for(std::vector<std::string>::const_iterator i_timing = parts.begin(); i_timing != parts.end(); ++i_timing)
        {
            Event event;
            event.EventID = (eventID ++);
            event.SequenceID = seqID;
            event.DimentionID = 0;
            event.time = atof(i_timing->c_str());
            event.marker = -1;
            seq.Add(event);
        }

        data.push_back(seq);
        ++seqID;
    }

    fin.close();
}
