import csv
import os

csv.Dialect.delimiter=','

with open ('../IRBI/truncated.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    reader._fieldnames=['number', 'species', 'phone', 'version']
    for row in reader:
        picture = '\'/home/corentin/Documents/IRBI/norecalib_scaled/'+row['number']+'\''
        dest = '\'/home/corentin/Documents/IRBI/'+row['species']+'/'+row['number']+'\''
        rep = '\'/home/corentin/Documents/IRBI/'+row['species']+'\''

        mkdir ='mkdir -p '+rep
        os.system(mkdir)

        cp = 'mv ' +picture +' ' +dest
        os.system(cp)
