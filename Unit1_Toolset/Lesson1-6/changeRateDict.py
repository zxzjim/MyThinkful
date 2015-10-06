changeRateDict = collections.defaultdict(float)
with open ('lecz-urban-rural-population-land-area-estimates_continent-90m.csv', 'rU') as inputFile:
    header = next(inputFile)
    print header
    for line in inputFile:
        line = line.rstrip().split(',')
        line[5] = float(line[5])
        line[6] = float(line[6])
        if line[1] == 'Total National Population':
            changeRateDict[line[0]] += (line[6]-line[5])/line[5]
            with open('national_population_change_rate.csv', 'w') as outputFile:



outputFile.write('continent, 2010-2100_population_change_rate\n')
for k,v in changeRateDict.iteritems():
    outputFile.write(k + ',' + str(v) + '\n')
