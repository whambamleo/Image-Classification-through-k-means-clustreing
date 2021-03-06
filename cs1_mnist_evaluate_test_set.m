%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.

close all;

predictions = zeros(200,1);
outliers = zeros(200,1);
distances = zeros(200,1);
norms = zeros(200,1);

% loop through the test set, figure out the predicted number
for i = 1:200
    testing_vector=test(i,:);

    % Extract the centroid that is closest to the test image
    [prediction_index, norms(i)]=assign_vector_to_centroid(testing_vector,centroids);
    predictions(i) = centroid_labels(prediction_index);
end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
% Find unusually dim/bright images
norm_dist = fitdist(norms, "Normal");
norm_iqr_bounds = icdf(norm_dist, [0.25, 0.75]);
norm_iqr = norm_iqr_bounds(2) - norm_iqr_bounds(1);
norm_iqr_bounds(1) = norm_iqr_bounds(1) - (norm_iqr * 1.5);
norm_iqr_bound = norm_iqr_bounds(2) + (norm_iqr * 1.5);

for index = 1:200
    if(norms(index) > norm_iqr_bound)
        outliers(index) = 1;
    end
end

% Find median values and proportion of non-zero pixels
medians = zeros(200, 1);
non_zero_proportion = zeros(200,1);

for index = 1:200
    image = test(index,1:784);
    non_zeros = image(image ~= 0);
    medians(index, 1) = median(non_zeros);
    non_zero_proportion(index,1) = width(non_zeros)/width(image);
end

% Find lower-bound outliers
median_threshold = 0.15;
median_dist = fitdist(medians, "Normal");
median_bound = icdf(median_dist, median_threshold);
non_zero_threshold = 0.75;
non_zero_dist = fitdist(non_zero_proportion, "Normal");
non_zero_bound = icdf(non_zero_dist, non_zero_threshold);

for index = 1:200
    if(medians(index, 1) < median_bound && non_zero_proportion(index, 1) > non_zero_bound)
        outliers(index) = 1;
    end
end

% Display outliers - FOR QUALITATIVE ANALYSIS ONLY
%{
figure;
colormap('gray');

outlier_vectors = test(outliers == 1, :);
plotsize = ceil(sqrt(height(outlier_vectors)));

for ind=1:height(outlier_vectors)
    current_vector = outlier_vectors(ind, 1:784);
    subplot(plotsize,plotsize,ind);
    imagesc(reshape(current_vector,[28 28])');
    title(strcat('Outlier ',num2str(ind)))
end
%}

%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
stem(outliers);
title('Outliers', 'FontSize', 18);    % Title and Label Axis
xlabel('Test Set Index', 'FontSize', 18);
ylabel('Flag', 'FontSize', 18);

%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
title('Predictions', 'FontSize', 18);    % Title and Label Axis
xlabel('Test Set Index', 'FontSize', 18);
ylabel('Label', 'FontSize', 18);

%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)

% Matrix representing actual vs. classified labels
%{
label_map = zeros(10, 10);
for index = 1:200
    label_map(correctlabels(index, 1) + 1, predictions(index, 1) + 1) =...
        label_map(correctlabels(index, 1) + 1, predictions(index, 1) + 1) + 1;
end

% Extracting statistics from matrix
num_misclassified = zeros(10, 1);
total_occurances = zeros(10, 1);
for outer = 1:10
    for inner = 1:10
        if(outer ~= inner)
            num_misclassified(outer, 1) = num_misclassified(outer, 1) + label_map(outer, inner);
        end
        total_occurances(outer, 1) = total_occurances(outer, 1) + label_map(outer, inner);
    end
end

for index = 1:10
    message = "Mis-classification Rate of " + (index - 1) + "'s: " +...
        num_misclassified(index, 1)*100/total_occurances(index, 1) + "%";
    disp(message)
end

% Display incorrect - FOR QUALITATIVE ANALYSIS ONLY
figure;
colormap('gray');

incorrect_vectors = test(correctlabels~=predictions, :);
plotsize = ceil(sqrt(height(incorrect_vectors)));

incorrect_labels = correctlabels(correctlabels~=predictions);
incorrect_predictions = predictions(correctlabels~=predictions);

for ind=1:height(incorrect_vectors)
    current_vector = incorrect_vectors(ind, 1:784);
    subplot(plotsize,plotsize,ind);
    imagesc(reshape(current_vector,[28 28])');
    title(strcat('Labeled ',num2str(incorrect_labels(ind)), ' | Prediction: ',num2str(incorrect_predictions(ind))));
end

% Display correct - FOR QUALITATIVE ANALYSIS ONLY
figure;
colormap('gray');

correct_vectors = test(correctlabels==predictions, :);
plotsize = ceil(sqrt(height(correct_vectors)));

for ind=1:height(correct_vectors)
    current_vector = correct_vectors(ind, 1:784);
    subplot(plotsize,plotsize,ind);
    imagesc(reshape(current_vector,[28 28])');
    title(strcat('Correct ',num2str(ind)));
end
%}

function [index, vec_norm] = assign_vector_to_centroid(data,centroids)
    k = height(centroids);
    closest_centroid = 0;
    distance_to_closest_centroid = Inf;
    
    for centroid_index = 1:k                                % Go through each centroid
        current_centroid = centroids(centroid_index,:);     % Get the current centroid
        distance = norm(data - current_centroid);           % Get the distance between the current digit and the current centroid
        if (distance < distance_to_closest_centroid)        % If the current centroid is closer than the previously closest centroid
            closest_centroid = centroid_index;              % Set closet_centroid to the current centroid's index
            distance_to_closest_centroid = distance;        % Set distance_to_closest_centroid to current_distance
        end
    end

    index = closest_centroid;                       % Return the index of the closest centroid
    vec_norm = norm(data);
end

