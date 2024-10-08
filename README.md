# Bitcoin Block Constructor

This project implements a Bitcoin block construction algorithm that selects transactions from a mempool to create a block that maximizes transaction fees while respecting block weight limits and transaction dependencies.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithm Overview](#algorithm-overview)
- [Performance Optimization](#performance-optimization)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- Parses mempool transactions from a CSV file
- Constructs a block that maximizes transaction fees
- Respects transaction dependencies and block weight limits
- Utilizes parallel processing for improved performance
- Includes comprehensive logging for debugging and monitoring
- Provides unit tests for key components

## Requirements

- Python 3.7+
- No external dependencies required

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/bitcoin-block-constructor.git
cd bitcoin-block-constructor
```

(Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

## Usage

1. Prepare your input file:

   - Create a CSV file named `mempool.csv` in the project root directory.
   - Each row should represent a transaction with the format:
     `txid,fee,weight,parent_txids`.

- Example:

  ```bash
  tx1,10,150,
  tx2,20,200,tx1
  tx3,30,250,tx1;tx2
  ```

2. Run the script:

   ```bash
   python bitcoin_block_constructor.py
   ```

3. Check the output:

   - The constructed block will be written to block.txt.
   - Each line in block.txt represents a transaction ID included in the block.

4. Review the logs:

   - The script will output logs to the console, providing information about the construction process.

## Project Structure

- `bitcoin_block_constructor.py`: Main script containing the block construction algorithm.
- `test_bitcoin_block_constructor.py`: Unit tests for the main components.
- `mempool.csv`: Input file containing mempool transactions (you need to create this).
- `block.txt`: Output file containing the constructed block (generated by the script).

## Algorithm Overview

- **Mempool Parsing**: The script reads transactions from `mempool.csv` and builds a dependency graph.
- **Parallel Block Construction**: Transactions are sorted by fee density and processed in parallel chunks.
- **Block Optimization**: The initial block is optimized by attempting to replace transactions with more profitable ones.
- **Finalization**: The final block is written to `block.txt`.

## Performance Optimization

    - The script uses parallel processing to improve performance on multi-core systems.
    - The number of processes used is automatically set to the number of CPU cores available.

## Testing

To run the unit tests:

```bash
python -m unittest test_bitcoin_block_constructor.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
