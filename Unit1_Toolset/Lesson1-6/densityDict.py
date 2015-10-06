totalNationDict = collections.defaultdict(float)
totalAreaDict = collections.defaultdict(float)
with open ('lecz-urban-rural-population-land-area-estimates_continent-90m.csv', 'rU') as inputFile:
    header = next(inputFile)
    print header
    for line in inputFile:
        line = line.rstrip().split(',')
        line[5] = float(line[5])
        if line[1] == 'Total National Population':
            totalNationDict[line[0]] += line[5]
totalNationDict
with open ('lecz-urban-rural-population-land-area-estimates_continent-90m.csv', 'rU') as inputFile:
    header = next(inputFile)
    print header
    for line in inputFile:
        line = line.rstrip().split(',')
        line[7] = float(line[7])
        if line[1] == 'Total National Population':
            totalAreaDict[line[0]] += line[7]
totalAreaDict
for k,v in totalNationDict.iteritems():
    densityDict[k] = v / totalAreaDict[k]

with open('national_population_density.csv', 'w') as outputFile:
    outputFile.write('continent, 2010_population_density\n')
    for k, v in densityDict.iteritems():
        outputFile.write(k + ',' + str(v) + '\n')