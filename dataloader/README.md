# Explanation

If the session files exist in the related directories, the program will use them. Otherwise, it will create new session files for each task, number of shots, and seed number combination.

<!-- Please note that the train samples file for the base task and all test sample text files are only created to show the sample files that the program will use. The program will ignore them later. -->

## References

Our data-loading related codes are based on the S3C (or TOPIC).

@inproceedings{kalla2022s3c,
  title={S3C: Self-supervised stochastic classifiers for few-shot class-incremental learning},
  author={Kalla, Jayateja and Biswas, Soma},
  booktitle={European Conference on Computer Vision},
  pages={432--448},
  year={2022},
  organization={Springer}
}

@article{Tao2020FewShotCL,
  title={Few-Shot Class-Incremental Learning},
  author={Xiaoyu Tao and Xiaopeng Hong and Xinyuan Chang and Songlin Dong and Xing Wei and Yihong Gong},
  journal={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
  pages={12180-12189}
}
