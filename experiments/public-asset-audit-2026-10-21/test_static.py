import unittest
from audit_vjepa_source import double_read_sites


class StaticAuditTests(unittest.TestCase):
    def test_detects_decoded_dataset_list_not_file_list(self):
        helper = 'def reader(files):\n return [pydicom.dcmread(x) for x in files]\n'
        self.assertEqual(len(double_read_sites(helper+'headers = reader([pydicom.dcmread(p) for p in files])')), 1)
        self.assertEqual(double_read_sites(helper+'headers = reader(files)'), [])


if __name__ == '__main__':
    unittest.main()
