# Qualitative Results of Semantic Segmentation on Video Sequences

This repository proposes _VQ-BNN inference_, a _temporal smoothing_ of BNN's recent predictions, in order to improve the computational performance of BNN. In addition, it also proposes _VQ-DNN_, which is a temporal smoothing of deterministic NN's predictions.
Experimental results show that the computational performance of VQ-BNN is almost the same as that of deterministic NN (DNN), and the predictive performance is comparable to or even superior to that of BNN. Similarly, the predictive performance of VQ-DNN is better than that of DNN.

This material provides predictive results and predictive uncertainties of DNN, VQ-DNN, BNN, and VQ-BNN on five different sequences.
According to these qualitative results, ___the predictive results of DNN and BNN are noisy___. Their classification results for an object change irregularly and randomly. In contrast, ___the predictive results of VQ-DNN and VQ-BNN are stabilized___. Their predictive results change smoothly. 


## Animated Predictive Results and Predictive Uncertainties


<table cellspacing="15" style="width:100%;">
  <tr>
    <th>Input</th>
    <th>DNN <div>(11 FPS)</div></th>
    <th>VQ-DNN <div>(10 FPS)</div></th>
    <th>BNN <div>(0.8 FPS)</div></th>
    <th>VQ-BNN <div>(9 FPS)</div></th>
  </tr>
  <tr>
    <th colspan="5" style="font-style:italic;">Seq 1</th>
  </tr>
  <tr>
    <td><img src="seq1/input-seq1.gif" style="width:100%;"></td>
    <td><img src="seq1/dnn-res-seq1.gif" style="width:100%;"></td>
    <td><img src="seq1/vqdnn-res-seq1.gif" style="width:100%;"></td>
    <td><img src="seq1/bnn-res-seq1.gif" style="width:100%;"></td>
    <td><img src="seq1/vqbnn-res-seq1.gif" style="width:100%;"></td>
  </tr>
  <tr>
    <td></td>
    <td><img src="seq1/dnn-unc-seq1.gif" style="width:100%;"></td>
    <td><img src="seq1/vqdnn-unc-seq1.gif" style="width:100%;"></td>
    <td><img src="seq1/bnn-unc-seq1.gif" style="width:100%;"></td>
    <td><img src="seq1/vqbnn-unc-seq1.gif" style="width:100%;"></td>
  </tr>
  <tr>
    <th colspan="5" style="font-style:italic;">Seq 2</th>
  </tr>
  <tr>
    <td><img src="seq2/input-seq2.gif" style="width:100%;"></td>
    <td><img src="seq2/dnn-res-seq2.gif" style="width:100%;"></td>
    <td><img src="seq2/vqdnn-res-seq2.gif" style="width:100%;"></td>
    <td><img src="seq2/bnn-res-seq2.gif" style="width:100%;"></td>
    <td><img src="seq2/vqbnn-res-seq2.gif" style="width:100%;"></td>
  </tr>
  <tr>
    <td></td>
    <td><img src="seq2/dnn-unc-seq2.gif" style="width:100%;"></td>
    <td><img src="seq2/vqdnn-unc-seq2.gif" style="width:100%;"></td>
    <td><img src="seq2/bnn-unc-seq2.gif" style="width:100%;"></td>
    <td><img src="seq2/vqbnn-unc-seq2.gif" style="width:100%;"></td>
  </tr>
  <tr>
    <th colspan="5" style="font-style:italic;">Seq 3</th>
  <tr>
    <td><img src="seq3/input-seq3.gif" style="width:100%;"></td>
    <td><img src="seq3/dnn-res-seq3.gif" style="width:100%;"></td>
    <td><img src="seq3/vqdnn-res-seq3.gif" style="width:100%;"></td>
    <td><img src="seq3/bnn-res-seq3.gif" style="width:100%;"></td>
    <td><img src="seq3/vqbnn-res-seq3.gif" style="width:100%;"></td>
  </tr>
  <tr>
    <td></td>
    <td><img src="seq3/dnn-unc-seq3.gif" style="width:100%;"></td>
    <td><img src="seq3/vqdnn-unc-seq3.gif" style="width:100%;"></td>
    <td><img src="seq3/bnn-unc-seq3.gif" style="width:100%;"></td>
    <td><img src="seq3/vqbnn-unc-seq3.gif" style="width:100%;"></td>
  </tr>
  <tr>
    <th colspan="5" style="font-style:italic;">Seq 4</th>
  <tr>
    <td><img src="seq4/input-seq4.gif" style="width:100%;"></td>
    <td><img src="seq4/dnn-res-seq4.gif" style="width:100%;"></td>
    <td><img src="seq4/vqdnn-res-seq4.gif" style="width:100%;"></td>
    <td><img src="seq4/bnn-res-seq4.gif" style="width:100%;"></td>
    <td><img src="seq4/vqbnn-res-seq4.gif" style="width:100%;"></td>
  </tr>
  <tr>
    <td></td>
    <td><img src="seq4/dnn-unc-seq4.gif" style="width:100%;"></td>
    <td><img src="seq4/vqdnn-unc-seq4.gif" style="width:100%;"></td>
    <td><img src="seq4/bnn-unc-seq4.gif" style="width:100%;"></td>
    <td><img src="seq4/vqbnn-unc-seq4.gif" style="width:100%;"></td>
  </tr>
  <tr>
    <th colspan="5" style="font-style:italic;">Seq 5</th>
  <tr>
    <td><img src="seq5/input-seq5.gif" style="width:100%;"></td>
    <td><img src="seq5/dnn-res-seq5.gif" style="width:100%;"></td>
    <td><img src="seq5/vqdnn-res-seq5.gif" style="width:100%;"></td>
    <td><img src="seq5/bnn-res-seq5.gif" style="width:100%;"></td>
    <td><img src="seq5/vqbnn-res-seq5.gif" style="width:100%;"></td>
  </tr>
  <tr>
    <td></td>
    <td><img src="seq5/dnn-unc-seq5.gif" style="width:100%;"></td>
    <td><img src="seq5/vqdnn-unc-seq5.gif" style="width:100%;"></td>
    <td><img src="seq5/bnn-unc-seq5.gif" style="width:100%;"></td>
    <td><img src="seq5/vqbnn-unc-seq5.gif" style="width:100%;"></td>
  </tr>
</table>

