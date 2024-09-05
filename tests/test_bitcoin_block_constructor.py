import unittest
from io import StringIO
from unittest.mock import patch
from bitcoin_block_constructor import MempoolManager, BlockConstructor, Transaction

class TestMempoolManager(unittest.TestCase):
    def setUp(self):
        self.mempool_manager = MempoolManager()

    def test_parse_mempool_csv(self):
        csv_content = StringIO("""tx1,10,100,
tx2,20,200,tx1
tx3,30,300,tx1;tx2
""")
        with patch('builtins.open', return_value=csv_content):
            self.mempool_manager.parse_mempool_csv('dummy.csv')

        self.assertEqual(len(self.mempool_manager.transactions), 3)
        self.assertIn('tx1', self.mempool_manager.transactions)
        self.assertIn('tx2', self.mempool_manager.transactions)
        self.assertIn('tx3', self.mempool_manager.transactions)

    def test_build_dependency_graph(self):
        self.mempool_manager.transactions = {
            'tx1': Transaction('tx1', 10, 100),
            'tx2': Transaction('tx2', 20, 200, ['tx1']),
            'tx3': Transaction('tx3', 30, 300, ['tx1', 'tx2'])
        }
        self.mempool_manager._build_dependency_graph()

        self.assertEqual(self.mempool_manager.transactions['tx1'].children, ['tx2', 'tx3'])
        self.assertEqual(self.mempool_manager.transactions['tx2'].children, ['tx3'])
        self.assertEqual(self.mempool_manager.transactions['tx3'].children, [])

class TestBlockConstructor(unittest.TestCase):
    def setUp(self):
        self.mempool_manager = MempoolManager()
        self.mempool_manager.transactions = {
            'tx1': Transaction('tx1', 10, 100),
            'tx2': Transaction('tx2', 20, 200, ['tx1']),
            'tx3': Transaction('tx3', 30, 300),
            'tx4': Transaction('tx4', 40, 400, ['tx3'])
        }
        self.mempool_manager.unconfirmed = set(self.mempool_manager.transactions.keys())
        self.block_constructor = BlockConstructor(self.mempool_manager)

    def test_construct_block(self):
        self.block_constructor.construct_block()
        self.assertEqual(len(self.block_constructor.block), 4)
        self.assertIn('tx1', self.block_constructor.block)
        self.assertIn('tx2', self.block_constructor.block)
        self.assertIn('tx3', self.block_constructor.block)
        self.assertIn('tx4', self.block_constructor.block)

    def test_block_weight_limit(self):
        # Add a transaction that would exceed the block weight limit
        self.mempool_manager.transactions['tx5'] = Transaction('tx5', 50, 3_000_000)
        self.mempool_manager.unconfirmed.add('tx5')
        
        self.block_constructor.construct_block()
        self.assertLess(self.block_constructor.block_weight, BlockConstructor.MAX_BLOCK_WEIGHT)

    def test_optimize_block(self):
        # Add a high-fee transaction that should replace others
        self.mempool_manager.transactions['tx5'] = Transaction('tx5', 100, 500)
        self.mempool_manager.unconfirmed.add('tx5')
        
        self.block_constructor.construct_block()
        initial_fees = self.block_constructor.block_fees
        
        self.block_constructor.optimize_block()
        self.assertGreater(self.block_constructor.block_fees, initial_fees)
        self.assertIn('tx5', self.block_constructor.block)

class TestTransaction(unittest.TestCase):
    def test_fee_density_calculation(self):
        tx = Transaction('tx1', 100, 200)
        self.assertEqual(tx.fee_density, 0.5)

    def test_fee_density_zero_weight(self):
        tx = Transaction('tx1', 100, 0)
        self.assertEqual(tx.fee_density, 0)

if __name__ == '__main__':
    unittest.main()