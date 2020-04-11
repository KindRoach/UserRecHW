from sklearn.model_selection import train_test_split

from tool.data_reader import all_ratings

ratings_train, ratings_test = train_test_split(all_ratings, random_state=42, train_size=0.8)
for file_name in [
    "out/SvdMF.csv",
    "out/ItemCF_10.csv",
    "out/UserCF_100.csv",
]:
    total_error = 0
    with open(file_name, 'r', encoding='utf-8') as f_in, \
            open(file_name + ".out.csv", 'w', encoding='utf-8') as f_out:
        i = 0
        for r in ratings_test:
            line = f_in.readline()
            cols = line.strip().split(' ')
            pre = float(cols[2])
            if not pre > 0:
                continue
            i += 1
            total_error += (r.rating - pre) ** 2
            f_out.write("%s\n" % (total_error / i))
            # print("%s" % (total_error / i))

