import csv
import os
import sys

if len(sys.argv) > 2:
    print("Trop de paramètre en entrée")
    print("Si le chemin contien des espaces, mettre des guillemets de part et autre")
    sys.exit(1)
elif len(sys.argv) <= 1:
    print("Bien mettre le chemin du fichier de répartition de l'IRBI")
    sys.exit(1)    
else:
    PATH=(sys.argv[1])

CSV_PATH=PATH + '/truncated.csv'
    DEST_PATH=PATH

    
csv.Dialect.delimiter=','

print(CSV_PATH)

if open (CSV_PATH) :
    print("Fichier pris en compte")
else :
    print("Problème dans l'ouverture du fichier")
    print("Veuillez vérifier le path du fichier")
    sys.exit(1)


mkdir ='mkdir -p ' + DEST_PATH
os.system(mkdir)

with open (PATH) as csvfile:
    reader = csv.DictReader(csvfile)
    reader._fieldnames=['number', 'species', 'phone', 'version']
    for row in reader:
        picture = '\'' + PATH + 'norecalib_scaled/' + row['number'] + '\''
        dest = '\'' + DEST_PATH + row['species'] + '/' + row['number']+'\''
        rep = '\'' + DEST_PATH + row['species']+'\''

        mkdir ='mkdir -p ' +rep
        os.system(mkdir)

        cp = 'mv ' + picture +' ' +dest
        os.system(cp)

rm = 'rm -Rf ' + '\'' + DEST_PATH + 'norecalib_scaled/' 