# Conditioned fully convolutional denoising autoencoder for multi-target NILM
by GSDPI *Supervision, Diagnosis and Knowledge Discovery in Engineering Processes Research Group*

https://gsdpi.edv.uniovi.es/

This repository includes the code needed to replicate the results of the paper:
García, D., Pérez, D., Papapetrou, P., Díaz, I., Cuadrado, A. A., Enguita, J. M., & Domínguez, M. (2025). Conditioned fully convolutional denoising autoencoder for multi-target NILM. Neural Computing and Applications, 37(17), 10491-10505. https://doi.org/10.1007/S00521-024-10552-0

### Abstract
Energy management requires reliable tools to support decisions aimed at optimising consumption. Advances in data-driven models provide techniques like Non-Intrusive Load Monitoring (NILM), which estimates the energy demand of appliances from total consumption. Common single-target NILM approaches perform energy disaggregation by using separate learned models for each device. However, the use of single-target systems in real scenarios is computationally expensive and can obscure the interpretation of the resulting feedback. This study assesses a conditioned deep neural network built upon a Fully Convolutional Denoising AutoEncoder (FCNdAE) as multi-target NILM model. The network performs multiple disaggregations using a conditioning input that allows the specification of the target appliance. Experiments compare this approach with several single-target and multi-target models using public residential data from households and non-residential data from a hospital facility. Results show that the multi-target FCNdAE model enhances the disaggregation accuracy compared to previous models, particularly in non-residential data, and improves computational efficiency by reducing the number of trainable weights below 2 million and inference time below 0.25 s for several sequence lengths. Furthermore, the conditioning input helps the user to interpret the model and gain insight into its internal behaviour when predicting the energy demand of different appliances.

### Funding
This work was funded in part by Ministerio de Ciencia e Innovación (MCNIN)/Agencia Estatal de Investigación (AEI) (MCIN/AEI/10.13039/501100011033) under Grant PID2020-115401GB-I00.
