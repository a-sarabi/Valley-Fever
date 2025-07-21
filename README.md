
<div align="center">

# Forecasting Coccidioidomycosis (Valley Fever) in Arizona: A Graph Neural Network Approach

[![arXiv](https://img.shields.io/badge/arXiv-2507.10014-b31b1b)](https://arxiv.org/abs/2507.10014)

<img width="750" alt="framework" src="misc/flowchart.jpg">

</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

**Authors:** Ali Sarabi¹, Arash Sarabi¹, Hao Yan¹, Beckett Sterner², Petar Jevtić³

¹ *School of Computing and Augmented Intelligence, Arizona State University*  
² *School of Life Sciences, Arizona State University*  
³ *School of Mathematical and Statistical Sciences, Arizona State University*

## Overview

This repository presents the first application of Graph Neural Networks (GNNs) to forecast Coccidioidomycosis (Valley Fever) cases in Arizona. Our hybrid model combines graph-based spatial modeling with Transformer-based temporal sequence processing to capture complex spatiotemporal dependencies between environmental factors and disease transmission.

### Key Results
- **13% MAPE** for 2-week forecasts
- **16% MAPE** for 4-week forecasts  
- **21% MAPE** for 8-week forecasts
- **23% MAPE** for 16-week forecasts
- **90% dimensionality reduction** while maintaining accuracy

## Methodology

### Model Architecture

The hybrid model consists of four key components:

1. **Graph Construction**: Environmental variables represented as nodes with edges weighted by Pearson correlation coefficients
2. **Feature Gate**: Dynamic selection of top 10% most informative features
3. **GATv2 Networks**: Graph attention mechanism for learning variable representations
4. **Transformer Encoder-Decoder**: Temporal sequence processing for multi-step forecasting

### Data Sources

- **Valley Fever Data**: Maricopa County surveillance data (2006-2024)
- **Environmental Variables**: Soil temperature, weather patterns, air quality (PM10), agricultural indicators
- **Temporal Resolution**: Weekly aggregation with up to 6-week lag features

## Results

| Forecast Horizon | MAPE | Top Predictors |
|------------------|------|----------------|
| 2-week | 13% | Soil temperature, humidity, PM10 |
| 4-week | 16% | Soil temperature, humidity, solar radiation |
| 8-week | 21% | Soil temperature, humidity, wind patterns |
| 16-week | 23% | Soil temperature, PM10, solar radiation |

**Key Finding**: 20" soil temperature maximum consistently ranked as the most important predictor across all forecast horizons.

## Usage

### Prerequisites
```bash
pip install tensorflow>=2.8.0 spektral pandas numpy matplotlib scikit-learn openpyxl
```

### Running the Model
```bash
# Feature analysis and selection
python "Step 1- GATv2 and Transformer_Simple Gating top k.py"

# Visualization
python "Step 2- Visualization.py"

# Forecasting
python "Step 3- Forecasting Model.py"
```

## Project Structure

```
Valley-Fever/
├── Data/                           # Datasets
├── Step 1- GATv2 and Transformer_Simple Gating top k.py  # Feature analysis
├── Step 2- Visualization.py        # Results visualization
├── Step 3- Forecasting Model.py    # Main forecasting model
├── gatv2_conv.py                   # GATv2 implementation
├── Other models/                   # Baseline models
└── Run Results/                    # Model outputs
```

## Scientific Contribution

This research advances environmental disease modeling through:

- **Methodological Innovation**: First GNN application to Valley Fever forecasting
- **Public Health Impact**: Reliable 16-week advance warning capability
- **Computational Efficiency**: 90% feature reduction with maintained accuracy
- **Automated Feature Discovery**: Eliminates manual feature engineering

## Citation

```bibtex
@misc{sarabi2025forecastingcoccidioidomycosisvalleyfever,
      title={Forecasting Coccidioidomycosis (Valley Fever) in Arizona: A Graph Neural Network Approach}, 
      author={Ali Sarabi and Arash Sarabi and Hao Yan and Beckett Sterner and Petar Jevtić},
      year={2025},
      eprint={2507.10014},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.10014}, 
}
```

## Contact

For questions about this research:
- **Ali Sarabi**: asarabi1@asu.edu
- **Arash Sarabi**: sarabi.arash@asu.edu
- **Hao Yan**: haoyan@asu.edu

*Arizona State University, Tempe, AZ 85281, USA*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
