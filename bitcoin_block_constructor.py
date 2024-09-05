"""
Bitcoin Block Constructor

This module implements a Bitcoin block construction algorithm that selects
transactions from a mempool to create a block that maximizes transaction fees
while respecting block weight limits and transaction dependencies.

The main components are:
1. MempoolManager: Parses and manages the mempool of transactions.
2. BlockConstructor: Constructs and optimizes a block from the mempool.
3. Parallel processing: Utilizes multiple CPU cores for faster block construction.

Usage:
    python bitcoin_block_constructor.py

The script reads transactions from 'mempool.csv' and outputs the constructed
block to 'block.txt'.

Requirements:
    Python 3.7+
"""

import csv
import time
import logging
from typing import Dict, List, Set
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    """
    Represents a single Bitcoin transaction.

    Attributes:
        txid (str): Unique identifier for the transaction.
        fee (int): Transaction fee in satoshis.
        weight (int): Transaction weight in weight units.
        parent_txids (List[str]): List of parent transaction IDs.
        children (List[str]): List of child transaction IDs.
        fee_density (float): Fee per weight unit, calculated on initialization.
    """

    txid: str
    fee: int
    weight: int
    parent_txids: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    fee_density: float = 0.0

    def __post_init__(self):
        """Calculate fee density after initialization."""
        self.fee_density = self.fee / self.weight if self.weight > 0 else 0

class MempoolManager:
    """
    Manages the mempool of transactions.

    This class is responsible for parsing the CSV file of transactions and
    building the dependency graph between transactions.

    Attributes:
        transactions (Dict[str, Transaction]): Dictionary of all transactions.
        unconfirmed (Set[str]): Set of unconfirmed transaction IDs.
    """

    def __init__(self):
        """Initialize an empty mempool."""
        self.transactions: Dict[str, Transaction] = {}
        self.unconfirmed: Set[str] = set()

    def parse_mempool_csv(self, file_path: str) -> None:
        """
        Parse the mempool CSV file and populate the transactions dictionary.

        Args:
            file_path (str): Path to the CSV file containing mempool transactions.
        """
        logger.info(f"Parsing mempool CSV file: {file_path}")
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 4:
                    txid, fee, weight, parent_txids = row
                    parent_txids = parent_txids.split(';') if parent_txids else []
                    tx = Transaction(txid, int(fee), int(weight), parent_txids)
                    self.transactions[txid] = tx
                    self.unconfirmed.add(txid)
                else:
                    logger.warning(f"Ignoring invalid row: {row}")

        logger.info(f"Parsed {len(self.transactions)} transactions")
        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build the transaction dependency graph."""
        logger.info("Building transaction dependency graph")
        for tx in self.transactions.values():
            for parent_txid in tx.parent_txids:
                if parent_txid in self.transactions:
                    self.transactions[parent_txid].children.append(tx.txid)
        logger.info("Dependency graph built successfully")

class BlockConstructor:
    """
    Constructs and optimizes a block from the mempool.

    This class is responsible for selecting transactions from the mempool
    to construct a block that maximizes fees while respecting the block
    weight limit and transaction dependencies.

    Attributes:
        MAX_BLOCK_WEIGHT (int): Maximum allowed block weight.
        mempool (MempoolManager): The mempool to construct the block from.
        block (List[str]): List of transaction IDs in the constructed block.
        block_weight (int): Total weight of the constructed block.
        block_fees (int): Total fees of the constructed block.
    """

    MAX_BLOCK_WEIGHT = 4_000_000

    def __init__(self, mempool: MempoolManager):
        """
        Initialize a BlockConstructor.

        Args:
            mempool (MempoolManager): The mempool to construct the block from.
        """
        self.mempool = mempool
        self.block: List[str] = []
        self.block_weight = 0
        self.block_fees = 0

    def construct_block(self) -> None:
        """
        Construct a block by selecting transactions from the mempool.

        This method sorts transactions by fee density and adds them to the
        block if they meet the criteria (respects weight limit and dependencies).
        """
        logger.info("Starting block construction")
        candidate_txs = sorted(
            self.mempool.transactions.values(),
            key=lambda tx: tx.fee_density,
            reverse=True
        )

        for tx in candidate_txs:
            if self._can_add_to_block(tx):
                self._add_to_block(tx)

        logger.info(f"Block constructed with {len(self.block)} transactions")

    def _can_add_to_block(self, tx: Transaction) -> bool:
        """
        Check if a transaction can be added to the block.

        Args:
            tx (Transaction): The transaction to check.

        Returns:
            bool: True if the transaction can be added, False otherwise.
        """
        if tx.txid in self.block or self.block_weight + tx.weight > self.MAX_BLOCK_WEIGHT:
            return False
        return all(parent_txid in self.block for parent_txid in tx.parent_txids)

    def _add_to_block(self, tx: Transaction) -> None:
        """
        Add a transaction to the block.

        Args:
            tx (Transaction): The transaction to add.
        """
        self.block.append(tx.txid)
        self.block_weight += tx.weight
        self.block_fees += tx.fee
        self.mempool.unconfirmed.remove(tx.txid)

    def optimize_block(self) -> None:
        """
        Optimize the block by replacing transactions to maximize fees.

        This method iteratively tries to replace transactions in the block
        with more profitable ones from the mempool.
        """
        logger.info("Starting block optimization")
        improvements = True
        optimization_rounds = 0
        while improvements:
            optimization_rounds += 1
            improvements = False
            for txid in list(self.mempool.unconfirmed):
                tx = self.mempool.transactions[txid]
                if self._can_replace_transactions(tx):
                    self._replace_transactions(tx)
                    improvements = True
        logger.info(f"Block optimization completed after {optimization_rounds} rounds")

    def _can_replace_transactions(self, new_tx: Transaction) -> bool:
        """
        Check if a new transaction can replace existing ones in the block.

        Args:
            new_tx (Transaction): The new transaction to consider.

        Returns:
            bool: True if the new transaction can replace existing ones, False otherwise.
        """
        if new_tx.txid in self.block:
            return False

        weight_to_remove = sum(self.mempool.transactions[txid].weight for txid in self.block 
                               if txid not in new_tx.parent_txids and txid not in new_tx.children)
        return (self.block_weight - weight_to_remove + new_tx.weight <= self.MAX_BLOCK_WEIGHT and
                new_tx.fee > sum(self.mempool.transactions[txid].fee for txid in self.block 
                                 if txid not in new_tx.parent_txids and txid not in new_tx.children))

    def _replace_transactions(self, new_tx: Transaction) -> None:
        """
        Replace existing transactions in the block with a new transaction.

        Args:
            new_tx (Transaction): The new transaction to add to the block.
        """
        to_remove = [txid for txid in self.block 
                     if txid not in new_tx.parent_txids and txid not in new_tx.children]
        
        for txid in to_remove:
            self.block.remove(txid)
            self.block_weight -= self.mempool.transactions[txid].weight
            self.block_fees -= self.mempool.transactions[txid].fee
            self.mempool.unconfirmed.add(txid)

        self._add_to_block(new_tx)

def process_chunk(chunk: List[Transaction], max_weight: int) -> List[str]:
    """
    Process a chunk of transactions for parallel block construction.

    Args:
        chunk (List[Transaction]): A subset of transactions to process.
        max_weight (int): Maximum allowed weight for this chunk.

    Returns:
        List[str]: List of transaction IDs selected for the block.
    """
    logger.info(f"Processing chunk of {len(chunk)} transactions")
    block = []
    weight = 0
    for tx in chunk:
        if weight + tx.weight <= max_weight:
            block.append(tx.txid)
            weight += tx.weight
    return block

def parallel_construct_block(mempool: MempoolManager, num_processes: int) -> List[str]:
    """
    Construct a block using parallel processing.

    Args:
        mempool (MempoolManager): The mempool to construct the block from.
        num_processes (int): Number of processes to use for parallel construction.

    Returns:
        List[str]: List of transaction IDs selected for the block.
    """
    logger.info(f"Starting parallel block construction with {num_processes} processes")
    sorted_txs = sorted(mempool.transactions.values(), key=lambda tx: tx.fee_density, reverse=True)
    chunk_size = len(sorted_txs) // num_processes
    chunks = [sorted_txs[i:i + chunk_size] for i in range(0, len(sorted_txs), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk, chunk, BlockConstructor.MAX_BLOCK_WEIGHT // num_processes)
                   for chunk in chunks]

        results = []
        for future in as_completed(futures):
            results.extend(future.result())

    logger.info("Parallel block construction completed")
    return results

def main():
    """
    Main function to run the Bitcoin block constructor.

    This function orchestrates the entire process of parsing the mempool,
    constructing and optimizing the block, and writing the results to a file.
    """
    start_time = time.time()
    logger.info("Starting Bitcoin block construction process")

    mempool_manager = MempoolManager()
    mempool_manager.parse_mempool_csv('mempool.csv')

    num_processes = multiprocessing.cpu_count()
    logger.info(f"Using {num_processes} processes for parallel construction")

    parallel_block = parallel_construct_block(mempool_manager, num_processes)

    block_constructor = BlockConstructor(mempool_manager)
    for txid in parallel_block:
        if block_constructor._can_add_to_block(mempool_manager.transactions[txid]):
            block_constructor._add_to_block(mempool_manager.transactions[txid])

    block_constructor.optimize_block()

    end_time = time.time()
    execution_time = end_time - start_time

    logger.info(f"Number of transactions in the block: {len(block_constructor.block)}")
    logger.info(f"Total block weight: {block_constructor.block_weight}")
    logger.info(f"Total fees collected: {block_constructor.block_fees}")
    logger.info(f"Execution time: {execution_time:.2f} seconds")

    with open('block.txt', 'w') as f:
        for txid in block_constructor.block:
            f.write(f"{txid}\n")

    logger.info("Block transactions written to 'block.txt'")

if __name__ == "__main__":
    main()