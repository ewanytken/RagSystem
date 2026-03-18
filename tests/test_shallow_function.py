import unittest

from app.logger import LoggerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        triplets_by_full_query = [{"dsf":12},{"sdfsdf":4324}, {"gh": 3434}]
        triplets_by_subject = [{"gh": 3434},{"cxv": 343}]
        triplets_by_full_query = []
        triplets_by_subject = []
        if triplets_by_full_query or triplets_by_subject:
            all_triplets = triplets_by_full_query + triplets_by_subject
            triplets = [dict(t) for t in set(frozenset(d.items()) for d in all_triplets)]
            print(triplets)
if __name__ == '__main__':
    unittest.main()