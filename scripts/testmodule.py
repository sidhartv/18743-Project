import logging

def test(a, b):
    fout = open("/afs/ece/usr/dstiffle/testfile.txt", "w")
    fout.write("Inside Python: args(a{} b{})".format(a, b))
    fout.close()

    return 0xcafe;

def init(arch_epath, weights_edirpath, cluster_epath):
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.DEBUG, filename="rnn_logfile.log")

    logger = logging.getLogger()
    logger.info("Successfully initialized")

    return [None, None]

def inferchk(iaddr, daddr, is_read, rnn):
    logger = logging.getLogger()
    return True

def infer(iaddr, daddr, is_read, rnn):
    return [1L]

def cleanup():
    pass
