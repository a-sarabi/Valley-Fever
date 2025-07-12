# Forecasting Coccidioidomycosis (Valley Fever) in Arizona: A Graph Neural Network Approach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

**Authors:** Ali Sarabi¹, Arash Sarabi¹, Hao Yan¹, Beckett Sterner², Petar Jevtić³

¹ *School of Computing and Augmented Intelligence, Arizona State University*  
² *School of Life Sciences, Arizona State University*  
³ *School of Mathematical and Statistical Sciences, Arizona State University*

## 🌵 Overview

This repository contains the first application of **Graph Neural Networks (GNNs)** to forecast Coccidioidomycosis (Valley Fever) cases in Arizona. Our hybrid model combines graph-based spatial modeling with Transformer-based temporal sequence processing to capture complex spatiotemporal dependencies between environmental factors and disease transmission.

### 🎯 Key Achievements
- **13% MAPE** for 2-week forecasts
- **16% MAPE** for 4-week forecasts  
- **21% MAPE** for 8-week forecasts
- **23% MAPE** for 16-week forecasts
- **90% dimensionality reduction** while maintaining accuracy
- **Automated feature discovery** eliminating manual feature engineering

## 📊 Background & Problem

Valley Fever is a fungal respiratory disease endemic to the arid southwestern United States, particularly Arizona. The disease has shown increasing incidence over the past two decades, making it a significant public health concern. The challenge lies in the complex interplay of environmental factors that influence disease transmission:

- **Soil conditions** (20" and 4" soil temperatures)
- **Weather patterns** (temperature, humidity, wind, precipitation)
- **Air quality** (PM10 concentrations)
- **Agricultural indicators** (evapotranspiration, heat units)

Traditional statistical models struggle to capture these intricate spatiotemporal dependencies and fail to adequately model the complex relationships between multiple environmental variables operating at different temporal scales.

## 🧠 Methodology

### Architecture Overview

Our hybrid model consists of four key components:

1. **Graph Construction**: Environmental variables are represented as nodes in a weighted graph, with edges weighted by Pearson correlation coefficients (correlations < 0.05 are set to zero)

2. **Feature Gate**: Dynamic feature selection mechanism that identifies and retains only the top 10% of most informative features, achieving 90% dimensionality reduction

3. **Graph Attention Networks (GATv2)**: Learn representations of environmental variables, dynamically weighting connections based on their contribution to disease prediction

4. **Transformer Architecture**: Encoder-decoder structure processes temporally ordered embeddings to model long-range temporal dependencies and generate multi-step forecasts

### Data Sources

- **Valley Fever Data**: Surveillance data from Maricopa County, Arizona (2006-2024)
- **Environmental Predictors**: 
  - Soil conditions (20" and 4" soil temperature measurements)
  - Atmospheric variables (temperature, humidity, wind, precipitation, solar radiation)
  - Agricultural indicators (evapotranspiration, heat units)
  - Air quality metrics (PM10 concentrations)
- **Temporal Resolution**: Weekly aggregation following MMWR epidemiological calendar
- **Lag Features**: Up to 6 weeks to capture delayed environmental effects

## 📈 Results

### Forecasting Performance

| Horizon | MAPE | Key Features |
|---------|------|-------------|
| 2-week  | 13%  | Soil temperature, humidity, PM10 |
| 4-week  | 16%  | Soil temperature, humidity, solar radiation |
| 8-week  | 21%  | Soil temperature, humidity, wind patterns |
| 16-week | 23%  | Soil temperature, PM10, solar radiation |

### Feature Importance Analysis

Through stability analysis using 100 different random seeds, we identified key environmental predictors:

**Most Consistent Predictors Across All Horizons:**
- **20" Soil Temperature - Max (0)** - Most reliable predictor
- **Relative Humidity - Min (-1)** - Critical moisture indicator
- **Daily Mean PM10 Concentration (-6)** - Air quality impact
- **Relative Humidity - Max (-1)** - Atmospheric moisture
- **Solar Radiation - Total (-2)** - Energy availability

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow>=2.8.0
pip install spektral
pip install pandas numpy matplotlib scikit-learn
pip install openpyxl
```

### Usage

1. **Feature Analysis & Selection**:
```bash
python "Step 1- GATv2 and Transformer_Simple Gating top k.py"
```

2. **Visualization**:
```bash
python "Step 2- Visualization.py"
```

3. **Forecasting**:
```bash
python "Step 3- Forecasting Model.py"
```

### Configuration

Update the data file path in the main scripts:
```python
data_file_path = "Data/Clean Processed Data2.csv"
```

## 📁 Project Structure

```
Valley-Fever/
├── Data/                                    # Raw and processed datasets
│   ├── Clean Processed Data2.csv          # Main processed dataset
│   ├── PM10 Data/                         # Air quality data
│   ├── Valley Fever Data/                 # Disease surveillance data
│   └── Weather Data MARICOPA/             # Environmental data
├── Step 1- GATv2 and Transformer_Simple Gating top k.py  # Feature analysis
├── Step 2- Visualization.py               # Results visualization
├── Step 3- Forecasting Model.py           # Main forecasting model
├── gatv2_conv.py                          # GATv2 implementation
├── Other models/                          # Baseline models
│   ├── Auto ARIMAX (Base Model).py       # ARIMAX baseline
│   └── Dynamic Training and Forecasting.py
└── Run Results/                           # Model outputs and analysis
```

## 🔧 Model Architecture Details

### Graph Processing Block
```python
def graph_processing_block(inputs, inp_lap, head_size, num_heads, dropout, horizon):
    # Feature gating for dimensionality reduction
    x = ImprovedFeatureGate(num_features=n_features, k_percent=0.1)(inputs)
    
    # Graph attention mechanism
    gc_output = GATv2Conv(
        channels=head_size,
        attn_heads=num_heads,
        dropout_rate=dropout,
        return_attn_coef=True
    )([x, inp_lap])
    
    return gc_output
```

### Transformer Encoder-Decoder
```python
def transformer_encoder_decoder_block(graph_output, decoder_inputs, d_model, num_heads):
    # Positional encoding
    enc_emb = PositionalEmbedding(max_sequence_length=200, d_model=d_model)(graph_output)
    
    # Encoder stack
    encoder_output = TransformerEncoder(d_model, num_heads, ff_dim=256)(enc_emb)
    
    # Decoder stack with attention
    decoder_output = TransformerDecoder(d_model, num_heads, ff_dim=256)(
        decoder_inputs, encoder_output
    )
    
    return decoder_output
```

## 📊 Performance Metrics

The model is evaluated using multiple metrics:
- **MAPE** (Mean Absolute Percentage Error)
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **CORR** (Correlation Coefficient)
- **RSE** (Root Squared Error)

## 🎨 Visualization

The project includes comprehensive visualization tools:
- **Prediction plots** for each forecast horizon
- **Feature importance heatmaps** across different time horizons
- **Attention mechanism visualizations**
- **Multi-horizon comparison charts**

## 🔬 Scientific Impact

This research represents a significant advancement in environmental disease modeling:

1. **Methodological Innovation**: First successful application of GNNs to Valley Fever forecasting
2. **Practical Impact**: Provides public health officials with reliable 16-week advance warning
3. **Computational Efficiency**: 90% dimensionality reduction while maintaining accuracy
4. **Automated Discovery**: Eliminates manual feature engineering requirements

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{sarabi2024forecasting,
  title={Forecasting Coccidioidomycosis (Valley Fever) in Arizona: A Graph Neural Network Approach},
  author={Sarabi, Ali and Sarabi, Arash and Yan, Hao and Sterner, Beckett and Jevti{\'c}, Petar},
  journal={Applied Soft Computing},
  year={2025},
  note={First application of Graph Neural Networks to Valley Fever forecasting}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Arizona Department of Health Services for Valley Fever surveillance data
- National Weather Service for environmental data
- EPA for air quality measurements
- Maricopa County for regional health data

## 📞 Contact

For questions about this research or collaboration opportunities, please contact:
- **Ali Sarabi**: asarabi1@asu.edu
- **Arash Sarabi**: sarabi.arash@asu.edu
- **Hao Yan**: haoyan@asu.edu
- **Beckett Sterner**: beckett.sterner@asu.edu
- **Petar Jevtić**: petar.jevtic@asu.edu

*Arizona State University, Tempe, AZ 85281, USA*

---

**Keywords**: Valley Fever, Coccidioidomycosis, Graph Neural Networks, Time Series Forecasting, Environmental Health, Public Health Surveillance, Machine Learning, Deep Learning 