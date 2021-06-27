# slm_scorer

> Scores a straight line mission GPX file

## Setup

- Make sure that `python3` and `pip3` is installed and available on the path (on Ubuntu: `sudo apt-get install python3 python3-pip`)
- You may wish to create a virtual environment to install the dependencies, for example with `python -m venv folder_name`
- Within the folder containing `requirements.txt`, run `python -m pip install -r requirements.txt` to install the dependencies

## Features

Scores a GPX file on how 'straight' the tracked route is. This can be based on
- Max distance from stright line
- Mean distance from straight line (weighted by progress toward target, or by time spent
  (note should 'clean' if using time weighting to handle pauses in track such as lunch stops, camping, etc)
- Root-mean-squared distance
- Linear regression (R^2 coefficient)
- Total distance travelled vs length of straight line
- Area between track and line (integral, trapezem method)
- Area between track and line (integral, shoelace method)
- Can also plot first and second derivatives of track to see how direction changes
  (can compare to target line which has first derivative = 0, can consider "straightness" by looking at
  second deivative since all straight lines have second derivative = 0)

## How to use

Can be called as a command line module

``` bash
python -m slm_scorer path/to/track.gpx
```
Can use `-h` command line argument to bring up the help page showing the options available.

To check everything worked in the first instance you can pass it the test GPX file with `python -m slm_scorer tests/test.gpx`. It should output the below.

```
(venv) $ python -m slm_scorer tests/test.gpx
  [INFO:slm_scorer]: Plot of latlon data saved as track.png
  [INFO:slm_scorer]: Plot of xy data saved as track_xy.png
  [INFO:slm_scorer]: Plot of rotated data saved as slm.png
  Max Distance is: 1635.0m
  Mean Distance (x weighted) is: 781.5m
  Mean Distance (time weighted) is: 801.0m
  RMS Distance (x weighted) is: 889.5m
  RMS Distance (time weighted) is: 902.5m
  R-Squared value is: 0.800059
  Distance travelled vs line is: 1.31 (15917.5m vs 12129.5m)
  Area between track and line (shoelace method) is: 31.3
  Area between track and line (trapezium method) is: 31.3
```

Or you can use pytest to run a suite of 40 tests with `python -m pytest` from within the folder containing the `tests` folder. It should output the below.

```
========================= test session starts =========================
platform win32 -- Python 3.9.0, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: C:\Users\toale\Documents\dev\slm_scorer\src
collected 40 items

tests\test_data.py ................                                                                                                                                                                                       [ 40%]
tests\test_init.py .........                                                                                                                                                                                              [ 62%]
tests\test_scores.py ...............                                                                                                                                                                                      [100%]

========================= 40 passed in 44.33s =========================
```

## Inspiration

- This package has been created in response to GeoWizard's 'Straight Line Mission' videos. See the video that inspired this package here <https://www.youtu.be/mElqGuzzwrs>
- Also see C. Burdell FitzGerald's [ScoreMyLine.com](https://www.scoremyline.com/).

## Licensing

This software is licensed under a joint commercial / open source license.

If you want to use this software to develop commercial web sites, tools, projects, or applications, the commercial license is the appropriate license. With this option, your source code is kept proprietary. To acquire a commercial License please [contact me](https://www.taylrr.co.uk/).

If you are creating an open source application under a license compatible with the GNU GPL license v3, you may use this software under the terms of the [GPLv3](LICENSE.txt).
