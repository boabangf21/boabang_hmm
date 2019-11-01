        function log_likelihood = log_likelihood(self, observations)
            B = self.state_likelihood(observations);
            log_likelihood = forward(self, observations, B);
        end