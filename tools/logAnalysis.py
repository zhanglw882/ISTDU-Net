# import


# srcFile = './logCtea.txt'
# srcFile = './logCtd.txt'
srcFile = './logCt.txt'
sf = open(srcFile, 'r')
aucs = []
for line in sf:
    if 'val_metrics' not in line:
        continue
    # print(line)
    auc = line.split(':')[-1]
    aucs.append(float(auc))
sf.close()
print(max(aucs))