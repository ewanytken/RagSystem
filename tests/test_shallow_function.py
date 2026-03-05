import unittest

from app.logger import LoggerWrapper
from app.respondent.local_model.transformer_wrapper import TransformerWrapper

logger = LoggerWrapper()

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_document_processing(self):
        import hashlib

        data = "Hello, world!"

        # Encode the string to bytes before hashing
        data_bytes = data.encode('utf-8')

        # Create a SHA-256 hash object
        hash_object = hashlib.sha256(data_bytes)

        # Get the hash value in hexadecimal format
        hex_digest = hash_object.hexdigest()

        print(hash_object)
        # Example Output: 315f5bdb78c9653a14589c7ad7255148565b20640c21e6490e56598c414963aa


if __name__ == '__main__':
    unittest.main()