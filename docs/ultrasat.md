
# ULTRASAT Mission
```
The Ultrasat mission is a proposed Israeli ultraviolet satellite project focused on comprehensive sky surveys. This repository serves as a central hub containing essential configurations, orbit details, and observation constraints vital for orchestrating the mission's operations. It includes specialized modules crafted for generating survey tiles across the sky, ensuring uniform coverage aligning with Ultrasat's Field of View (FOV) of 204 square degrees.
```
## Dorado Scheduling Skygrid  to Create **ultrasat.tess** tesselation

### Installing **[dorado-scheduling]**

```
pip install git+https://github.com/nasa/dorado-scheduling
```

### Then Run

```
  dorado-scheduling-skygrid --area "204 deg2" --output ULTRASAT.tess --method healpix**
```














[dorado-scheduling]: https://github.com/nasa/dorado-scheduling.git
