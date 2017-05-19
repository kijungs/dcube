D-Cube: Dense-Block Detection in Terabyte-Scale Tensors
========================
**D-Cube (Disk-based Dense-block Detection)** is an algorithm for detecting dense subtensors in tensors.
**D-Cube** has the following properties:
 * *scalable*: **D-Cube** handles large data not fitting in memory or even on a disk.
 * *fast*: Even when data fit in memory, **D-Cube** outperforms its competitors in terms of speed.
 * *accurate*: **D-Cube** gives high accuracy in real-world data as well as theoretical accuracy guarantees.

Datasets
========================
The download links for the datasets used in the paper are [here](http://www.cs.cmu.edu/~kijungs/codes/dcube/)

Building and Running D-Cube
========================
Please see [User Guide](user_guide.pdf)

Running Demo
========================
For demo, please type 'make'

Reference
========================
If you use this code as part of any published research, please acknowledge the following paper.
```
@inproceedings{shin2017dcube,
  title={D-cube: Dense-block detection in terabyte-scale tensors},
  author={Shin, Kijung and Hooi, Bryan and Kim, Jisu and Faloutsos, Christos},
  booktitle={Proceedings of the Tenth ACM International Conference on Web Search and Data Mining},
  pages={681--689},
  year={2017},
  organization={ACM}
}
```
