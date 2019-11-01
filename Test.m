predicted_word_labels = vocabulary.train_test(audio_signals', word_labels', audio_signals');
 
 Missclassification = 0;
 for i = 1:length(word_labels)
     fact  = word_labels(i);
     guess = predicted_word_labels(i);
     if ~isequal(fact, guess)
         Missclassification = Missclassification + 1;
         display(sprintf('Inccuracy_recognition  %d: Predicted %s, but was %s.', Inccuracy_recognition , char(guess), char(fact)))
     end
 end
 
 cvmcr = Inccuracy_recognition  / length(word_labels)