### Jobs

- `qsub` : Submit a job
- `qstat` : Stats


`qstat -g c`


option -xd
#### Submit a job

Batch
```
qsub -jc pcc-normal -cwd ./-jc pcc-normal
```


Interactivate
```
qrsh -ac d=nvcr-tensorflow-1712  -jc gpu-container_g1_dev
./fefs/opt/dgx/env_set/nvcr-tensorflow-1712.sh
./fefs/opt/dgx/env_set/common_env_set.sh
```




### Queues

    gpu-container_g1.168h                                                    X 
    gpu-container_g1.24h                                                     X 
    gpu-container_g1.72h                                                     X 
    gpu-container_g1.default                                                 X 
    gpu-container_g1_dev.default                                          X 
