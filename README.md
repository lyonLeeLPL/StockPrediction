<h1 align="center"> OBS! </h1>
This is NOT buying recommendations for trading real stocks and one should **NOT** trade real money based on the results here! 

<h1 align = "center"> Checklist </h1>
<br>

| Checkpoint                                              | Status |
| ------------------------------------------------- | ----   |
| :black_circle: <input type="checkbox" disabled checked /> Datacollection  |  :heavy_check_mark:    |
| :black_circle: <input type="checkbox" disabled  checked/>  Initial feature engineering |  :heavy_check_mark:    |
| :black_circle: <input type="checkbox" disabled  checked/> Visualization |  :heavy_check_mark:    |
| :black_circle: <input type="checkbox" disabled  checked/> Implement basic model |  :heavy_check_mark:    |
| :black_circle: <input type="checkbox" disabled  checked/> Initial predictions |  :heavy_check_mark:    |
| :black_circle: <input type="checkbox" disabled  checked/> Multi-feature predictions |   :clock930:   |
| :black_circle: <input type="checkbox" disabled  checked/> Improved visualization |  :clock930:    |
| :black_circle: <input type="checkbox" disabled  checked/> Build own transformer model |  :clock930:    |
| :black_circle: <input type="checkbox" disabled  checked/> Start build dash app |  :clock930:    |
| :black_circle: <input type="checkbox" disabled  checked/> Deploy dash app |  :clock930:    |
| :black_circle: <input type="checkbox" disabled  checked/> Real-time update |  :clock930:    |


<hr>
           


<h1 align="center"> StockPrediction </h1>

Hobby project in NLP with numerical time-series - stock market prediction based on the adjusted closing price, monthly gain and correlation among different industry-based companies from self build transformer model (to be done). Collecting the data (using pandas datareader from [yahoo finance](https://finance.yahoo.com/)). Initial predictions are done on just the [adjusted closing price](https://www.investopedia.com/terms/a/adjusted_closing_price.asp) with a LSTM model with a linear bottleneck. The loss being used is MSE: 
<br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{L}_{MSE}(\hat{y})&space;=&space;\frac{1}{N}\sum_{i=1}^N\left(y_i&space;-&space;\hat{y}_i\right)^2" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?\mathcal{L}_{MSE}(\hat{y})&space;=&space;\frac{1}{N}\sum_{i=1}^N\left(y_i&space;-&space;\hat{y}_i\right)^2" class="center" title="\mathcal{L}_{MSE}(\hat{y}) = \frac{1}{N}\sum_{i=1}^N\left(y_i - \hat{y}_i\right)^2" /></a>. 

<br>

Aim of the project is to build a [transformer model](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) from scratch and comparing its result with the LSTM network and doing so by ensemble results from different multi-feature predictions from different time periods. 

![transformer architecture](https://lilianweng.github.io/lil-log/assets/images/transformer.png?raw=true)

<br>
<hr>
<h1 align = "center"> Current status </h1>
<h3 align = "center"> Visualization and basic feature engineering </h3>
Collection and visualization of data from arbitrary timespan:

![GOOG stock example](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/goog_stocks_ex.png?raw=true)

<br>

![Goog histogram example](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/goog_stocks_hist_ex.png?raw=true)

<br>

Using [plotly](https://plotly.com/python/) for python to make interactive plots in order to zoom and move in the graph:

![MSFT stock example](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/msft_stocks_ex.png?raw=true)

![MSFT stock example](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/msft_stocks_square_ex.png?raw=true)

![MSFT stock example](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/msft_stocks_zoom_ex.png?raw=true)

<br>
<br>

Also, comparison of arbitrary number of stocks and visualize the correlation between them are also available:

![Comparison graphical](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/graph_corr.png?raw=true)

![Comparison graphical](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/corr_matrix.png?raw=true)

<hr>

<h3 align = "center"> Initial Predictions </h3>
The intial predictions on the adjusted closing price after implemented a LSTM model built from the Keras framework:

![Initial predictions](https://github.com/olof98johansson/StockPrediction/blob/main/demonstration_images/goog_pred_ex.png?raw=true)





