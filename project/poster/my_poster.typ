#import "@preview/cetz:0.2.2"
#import "@preview/unify:0.5.0": num,qty,numrange,qtyrange

#import "poster.typ": *

#show: poster.with(
  size: "36x24",
  title: "Analyzing Sea Surface Temperature Using Autoencoders",
  authors: "Louis Kamp Eskildsen - S203977",
  departments: "DTU compute",
  univ_logo: "./images/DTU_logo.svg",
  footer_text: "02671 Data-Driven Methods for Computational Science and Engineering",
  footer_url: "May 2024",
  footer_email_ids: "s203977@dtu.dk",
  footer_color: "990000",

  // Additional Parameters
  // =====
  // For 3-column posters, these usually DO NOT require any adjustments.
  // However, they are important for 2-column posters.
  // Refer to ./examples/example_2_column_18_24.typ for an example.
  // Defaults are commented on the right side

  keywords: (), // default is empty
  num_columns: "3", // 3
  univ_logo_scale: "100", // 100%
  univ_logo_column_size: "1.6", // 10in
  title_column_size: "20", // 20in
  title_font_size: "48", // 48pt
  authors_font_size: "36", // 36pt
  footer_url_font_size: "30", // 30pt
  footer_text_font_size: "40", // 40pt
)


= SST and The Climate

Oceans cover 70% of the Earths surface and is responsible for driving many of the vital parts of the climate. The temperature of the Oceans can have a major effect on local climate as seen by the Gulf Stream that lifts and regulates the temperature in western Europe by bringing hot water from the Gulf of Mexico. Another example of effects caused by sea surface temperature (SST) is El niño. El niño is a 
phenomenon located in the Pacific out from the coast of South America where hot water rise to the surface. When El niño occurs it is known to cause higher than normal temperatures around the world. El niño has also been linked to droughts and extreme rainfall. Therefore, predicting when an El niño cycle begins and ends helps in predicting the global climate. In this project it has been shown that the global SST can partially be encoded by an autoencoder and forecasted using a DMD model.

#figure(
  image("./images/svd_el_nino.svg", width: 93%),
  caption: [This plot shows the El niño as a mode of the singular value decomposition.]
) <El_nino>

Previous studies have shown that a partial prediction of SST can be done with dynamic mode decomposition (DMD) @ResDMD. This project shows that the DMD algorithm can be used in combination with an autoencoder to simplify computational complexity by reducing the number of parameters to model. The SST dataset used in this project comes from NOAA Optimum Interpolation and gives the average monthly SST from 1981 to 2022 with 44.219 datapoints each month @SST.

#figure(
  image("./images/svd_analysis.svg", width: 90%),
  caption: [This figure show the SVD singular values and the cumulative explained variance of the SVD modes.]
) <SVD_analysis>

There are multiple ways of encoding SST in a latent space. A commonly used method is the singular value decomposition (SVD). To estimate the size of the latent space of the autoencoder, an SVD analysis is carried out. @SVD_analysis shows the power of each sigular value and the cumulative explained variance captured by the SVD modes. The third of these modes can be seen in @El_nino and shows the carceraristics of EL niño phenomenon. The power of each singular value is seen to be decreasing before leveling off at approximately the 500'th component. This tells us that the underlying dimensionality of the dataset is around 500. Basing the autoencoder on a latent size of this is therefore a good starting point.

#bibliography("bib.yml", title: none)

#linebreak()

= Autoencoder for SST

To encode the SST an autoencoder based on a feedforward neural network has been implemented in PyTorch.
The autoencoder will be evaluated with the mean squared error. Let the operator $cal(N)$ represent the transformation from the true SST to the latent space by the neural network . Then latent mapping is then given by: $y_i = cal(N) x_i$ and the inverse by $x_i approx cal(N)^(-1) y_i$. By minimizing the MSE on $x_i approx cal(N)^(-1) cal(N) x_i$, the neural network learns how encode and decode the SST dataset.

The advantages of using an autoencoder over the latent space of the SVD is that the autoencoder might give a better representation of the non-linear sates since the feedforward neural network is inherently non-linear due to its activation functions.

== The architecture of the model
  
The autoencoder is composed of two large hidden layers as well as one layer in the middel that represents the latent space. ReLu is used as the activation function between the input and hidden layer as well as the hidden and output layer. To prevent overfitting a dropout layer is placed between the input and hidden layer during training.

#figure(
  cetz.canvas({
    import cetz.draw: *
    scale(x:100%, y:100%)
    let a = 1;
    let input_nodes = 6;
    let hidden_nodes = 4;
    let latent_nodes = 2;

    let input_nodes_content = ($x_1$, $x_2$, $x_3$, $x_(N-2)$, $x_(N-1)$, $x_(N)$)
    let hidden_nodes_content = ($x_1$, $x_2$, $x_(N-1)$, $x_(N)$)
    let latent_nodes_content = ($x_1$, $x_(N)$)

    set-style(stroke: none, radius:0.75)
    
    let b1 = (input_nodes - input_nodes/2)*2;
    let b2 = (hidden_nodes - hidden_nodes/2)*2;
    let b3 = (latent_nodes - latent_nodes/2)*2;
    let b4 = (hidden_nodes - hidden_nodes/2)*2;
    let b5 = (input_nodes - input_nodes/2)*2;

    for input_node in range(input_nodes) {

      for hidden_node in range(hidden_nodes) {

        line((0,b1 - 2*input_node), (4,b2 - 2*hidden_node), stroke: gray)
      }

      circle((0,b1 - 2*input_node), fill: rgb(216, 149, 218))
      content((), [#input_nodes_content.at(input_node)])

    }

    circle((0,1.05))
    content((), [$dots.v$])

    content((1,b1 + 2), [Input size = 44 219])

    for hidden_node in range(hidden_nodes) {

      for latent_node in range(latent_nodes) {
        line((4,b2 - 2*hidden_node), (8,b3 - 2*latent_node), stroke: gray)
      }

      circle((4,b2 - 2*hidden_node), fill: rgb(214, 88, 159) )
      content((), [#hidden_nodes_content.at(hidden_node)])
    }

    circle((4,1.05))
    content((), [$dots.v$])
    content((5,b2 +2 ), [Hidden size = 10 000])

    for latent_node in range(latent_nodes) {

      for hidden_node in range(hidden_nodes) {
        line((8,b3 - 2*latent_node), (12,b4 - 2*hidden_node), stroke: gray)
      }

      circle((8,b3 - 2*latent_node), fill: rgb(210, 0, 98) )
      content((), [#latent_nodes_content.at(latent_node)])
    }

    circle((8,1.05))
    content((), [$dots.v$])
    content((8,b3 + 2), [Latent size = 500])

    for hidden_node in range(hidden_nodes) {

      for input_node in range(input_nodes) {
        line((12, b4 - 2*hidden_node), (16, b5 - 2*input_node), stroke: gray)
      }
      

      circle((12,b4 - 2*hidden_node), fill: rgb(214, 88, 159))
      content((), [#hidden_nodes_content.at(hidden_node)])
    }

    circle((12,1.05))
    content((), [$dots.v$])

    for input_node in range(input_nodes) {
      circle((16,b5 - 2*input_node), fill: rgb(216, 149, 218))
      content((), [#input_nodes_content.at(input_node)])
    }

    circle((16,1.05))
    content((), [$dots.v$])
    
  }),
  caption: [The architecture of the autoencoder network.]
)


== Training and Tuning With Validation Data
The SST dataset is split into three partisions: 70%: training, 20%: validation, 10%: testing. 

The process of training the autoencoder will in this project happen in two stages. (1) Initial training: In this step the training dataset will be used to train the encoding and decoding of the SST into the latent space.
(2) Tuning: This step will use the validation dataset as input to optimize the models ability to encode, forecast and decode the data. It was found that the Adam optimizer worked best for initial training and SGD optimizer for the tuning.
#figure(
  image("./images/model_training_losses.svg", width: 90%),
  caption: [ (Training): The final train loss: $0.13$, validation loss: $0.28$. #linebreak()
  (Tuning): The final validation loss: $0.44$, test loss: $ 0.50$.]
) <training>

#text(fill: white,[This needs to be here adljfldjsbbdsjlgbsldjgb dspighsldjg sdilghsldigh lisdhgsldihg ])
= Forecasting SST in latent space with DMD
To forecast the SST the DMD method will be used.
In order to use the DMD method the encoded training data snapshots are placed into two matrixes:
$ bold(W)' = [bold(y)_i], i=2 dots.h n, quad "and" quad bold(W) = [bold(y)_i], i=1 dots.h (n-1) "." $
The DMD matrix $bold(A)$ is found by using the pseudoinverse of $bold(W)$:
$ bold(W)' = bold(A) bold(W) arrow.r bold(A) = bold(W)' bold(W)^(dagger) $
By solving for $bold(A)$ a linear model between one snapshot and the next has been made. To predict the SST in the latent space the following model can be used:
$ x_(i) approx cal(N)^(-1) bold(A)^i cal(N) x_(0)$. The figures below show the forecasted SST, which are predicted from February 2018, the first snapshot of the testing dataset. The prediction map shows an El niño phenomenon happening in September of 2021, and the error map shows this prediction to be wrong since the real El niño phenomenon happened in 2019. Comparing the model RMSE map and SST STD map also shows that the model has trouble in predicting where SST varies.
#figure(
  [
    #image("./images/model_el_nino_prediction.svg", width: 100%)
    #place(
      top + right,
      dx: 0.7in,
      dy: 0.25in,
      image("./images/qr.svg", width: 11%)
    )
    #place(
      top + right,
      dx: 0.63in,
      dy: 1.35in,
      [
        #set text(size: 0.5em, style: "italic")
        (Scan for animation)
      ]
    )
  ],
  caption: [This figure show how the models predicts an El niño event in Marts of 2020.]
) <mean_error>

#figure(
  image("./images/model_evaluation.svg", width: 100%),
  caption: [This figure show that the RMSE is highly correlated with the STD of the SST.]
) <std_error>

= Conclusion

These results suggest that an autoencoder is capable of encoding the SST data into a latent space, and the DMD algorithm is able to pick up on some of the dynamics of the system but is not capable of accurately long term forecasting. @time_error shows that the performance of the autoencoder is slightly worse than using the latent space of the SVD in combination with DMD. The predictions might be improved by replacing DMD with a more flexible method such as a RNN or LSTM architecture.

#figure(
  image("./images/model_prediction_error.svg", width: 90%),
  caption: [(Left): The plot shows the final loss for a model with a given latent space. The plots confirms that the optimal latent space is around 500. (Right): The mean squared error of the model as it predicts the SST from the first snapshot in the test dataset.]
) <time_error>