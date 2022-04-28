---
title: 'EzFlow: A modular PyTorch library for optical flow estimation using neural networks'
tags:
    - 'Optical Flow'
    - 'Neural Networks'
    - 'Computer Vision'
    - 'Machine Learning'
    - 'Python'
    - 'PyTorch'
authors:
    - name: Neelay Shah^[Corresponding author]
      orcid: 0000-0001-7950-1956
      affiliation: 1
    - name: Prajnan Goswami
      affiliation: 2
    - name: Huaizu Jiang
      affiliation: 2 
affiliations:
    - name: Birla Institute of Technology and Science, Pilani - Goa Campus, India
      index: 1
    - name: Northeastern University, USA
      index: 2
date: 28 April, 2022
bibliography: paper.bib
---

# Summary

`EzFlow` is a modular library for optical flow estimation using neural networks. It is written in Python and developed using the `PyTorch` [@NEURIPS2019_9015] framework for machine learning. `EzFlow` serves three purposes: (i) it contains modular implementations of established neural network architectures for optical flow estimation for off-the-shelf usage; (ii) it facilitates training of optical flow models by providing dataloaders for popular datasets and a configurable training pipeline; and (iii) it enables users to easily experiment with designing new model architectures by combining modular components and custom new layers using a registry and configuration system. The goal of this library is for users (researchers / practitioners) to be able to seamlessly work with neural networks for the optical flow estimation task.

# Statement of need

Over the past decade, the use of neural networks for the computer vision task of optical flow estimation has been on the rise. A sizeable number of research papers have been published on the topic, starting from **FlowNet** [@7410673] which was the one of the first ones to successfully demonstrate the use of neural networks for this task, to **RAFT** [@10.1007/978-3-030-58536-5_24] which is one of the best performing models, and finally to the recent work using transformer-based models for the task, **FlowFormer** [@https://doi.org/10.48550/arxiv.2203.16194].