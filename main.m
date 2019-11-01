clc
clear
close all

[audio_signals word_labels] = load_audio_from_folder('recordings');

display(sprintf('Loaded a total of %d audio signals for the following words:', length(audio_signals)))
display(unique(word_labels))

vocabulary = Vocabulary;
classf=@vocabulary.train_test;
crossval('cvmcr', audio_signals', word_labels', 'predfun', classf, 'kfold', 5)

predicted_word_labels = vocabulary.train_test(audio_signals', word_labels', audio_signals');

 Missclassification= 0; % missing observations
 for i = 1:length(word_labels)
     fact  = word_labels(i);
     guess = predicted_word_labels(i);
     if ~isequal(fact, guess)
         Missclassification = Missclassification + 1;
         display(sprintf('Missclassification %d: Predicted %s, but was %s.', Missclassification, char(guess), char(fact)))
     end
 end
 
 cvmcr = Missclassification / length(word_labels) % rate of missclassification