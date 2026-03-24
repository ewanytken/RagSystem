import unittest

from app.logger import LoggerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        ent = [{'entity': 'authorization token', 'label': 'parameter', 'score': 0.6607719659805298},
             {'entity': 'APIKEY', 'label': 'parameter', 'score': 0.657850980758667},
             {'entity': 'UUID', 'label': 'parameter', 'score': 0.7350522875785828},
             {'entity': 'Authorization', 'label': 'parameter', 'score': 0.7028675079345703},
             {'entity': 'HuggingFace', 'label': 'framework', 'score': 0.9158324003219604},
             {'entity': 'HuggingFace', 'label': 'framework', 'score': 0.9011632800102234},
             {'entity': 'HuggingFace', 'label': 'framework', 'score': 0.9437978267669678},

             {'entity': 'HuggingFace', 'label': 'framework', 'score': 0.9158324003219604},
             {'entity': 'HuggingFace', 'label': 'framework', 'score': 0.9011632800102234},
             {'entity': 'HuggingFace', 'label': 'framework', 'score': 0.9437978267669678}]


        unique_dicts = [dict(s) for s in set(frozenset(d.items()) for d in ent)]
        print(unique_dicts)

        print(len(unique_dicts))

        seen_ids = set()
        unique_data = []

        for d in ent:
            if d['entity'] not in seen_ids:
                unique_data.append(d)
                seen_ids.add(d['entity'])

        print(len(unique_data))
    if __name__ == '__main__':
        unittest.main()