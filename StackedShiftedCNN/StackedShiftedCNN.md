# Stacked Shiftedn CNN layers 

I attempted to create a new type of CNN layer in which the convolution filters are shifted by a certain amount to cover all aspects of the image with non-one strides. Typically, all filters in a CNN layer with a non-one stride are applied to the same segments of the image. In this attempt, I tried to shift the filters by a certain amount to cover all aspects of the image. The idea was to create a new type of CNN layer that can capture more information from the image. 

#### Testing:

Model does not do well in compared to traditional methods. Further investigation is needed to understand why this model does not perform well. There is still potential in this idea.