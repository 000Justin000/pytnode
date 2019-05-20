for i in {2017..2018}
do
    for j in {1..12}
    do
        curl -H "Accept: application/json" -H "Content-Type: application/json" -X GET https://api.nytimes.com/svc/archive/v1/$i/$j.json?api-key=UgPbbIQNKGevt7mcKRMD56P2f7FwwtOc > nyt$i$j
        sleep 10s
    done
done
