def test(a, b):
    fout = open("/afs/ece/usr/dstiffle/testfile.txt", "w")
    fout.write("Inside Python: args(a{} b{})".format(a, b))
    fout.close()

    return 0xcafe;
