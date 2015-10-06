import collections
changeDict = collections.defaultdict(int)
with open ('lecz-urban-rural-population-land-area-estimates_continent-90m.csv', 'rU') as inputFile:
    header = next(inputFile)
    print header
    for line in inputFile:
        line = line.rstrip().split(',')
        line[5] = int(line[5])
        line[6] = int(line[6])
        if line[1] == 'Total National Population':
            changeDict[line[0]] += line[6]-line[5]