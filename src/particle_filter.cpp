#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 500;  // TODO: Set the number of particles
  std::default_random_engine gen;   // generates pseudo-random numbers

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Create normal distribution for x, y and theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; ++i){
    Particle particle;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;   // generates pseudo-random numbers

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  for (int i = 0; i < num_particles; ++i){
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    if (yaw_rate > 0.001){
      x += (velocity / yaw_rate) * (sin(theta+yaw_rate*delta_t) - sin(theta));
      y += (velocity / yaw_rate) * (cos(theta) - cos(theta+yaw_rate*delta_t));
      theta += yaw_rate * delta_t;
    }
    // if yaw_rate < 0.001, the car probably moving straights
    else{
      x += velocity * cos(theta) * delta_t;
      y += velocity * sin(theta) * delta_t;
    }

    // Create normal distribution for x, y and theta
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */


  for (int i = 0; i < observations.size(); ++i){
    // loop through each observation
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;
    for (int j = 0; j < predicted.size(); ++j){
       // loop through each predicted measurement
      double pre_x = predicted[j].x;
      double pre_y = predicted[j].y;
      double smallest_dist = 100.0;
      double distance = dist(obs_x, obs_y, pre_x, pre_y);
      if (smallest_dist > distance){
        smallest_dist = distance;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double x_obs, y_obs, map_x, map_y, distance;

  for (int i = 0; i < num_particles; ++i){
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    vector<LandmarkObs> predicted;
    vector<LandmarkObs> transformed_obs;

    if (i == 0){
      for (int j = 0; j < observations.size(); ++j){
        // transform each observation marker from the vehicle's coordinates to the map's coordinates
        x_obs = cos(theta) * observations[j].x - sin(theta) * observations[j].y + x;
        y_obs = sin(theta) * observations[j].x + cos(theta) * observations[j].y + y;
        LandmarkObs tobs;
        tobs.x = x_obs;
        tobs.y = y_obs;
        transformed_obs.push_back(tobs);
      }
    }
    for (int m = 0; m < map_landmarks.landmark_list.size(); ++m){
      map_x = map_landmarks.landmark_list[m].x_f;
      map_y = map_landmarks.landmark_list[m].y_f;
      distance = dist(x, y, map_x, map_y);
      if (distance <= sensor_range){
        LandmarkObs predict_obs;
        predict_obs.x = map_x;
        predict_obs.y = map_y;
        predict_obs.id = map_landmarks.landmark_list[m].id_i;
        predicted.push_back(predict_obs);
      }
    }

    dataAssociation(predicted, transformed_obs);

    double mu_x;
    double mu_y;
    double prob;
    double w{1.0};
    for (int b = 0; b < transformed_obs.size(); ++b){
      particles[i].associations.push_back(transformed_obs[b].id);
      for (int p = 0; p < predicted.size(); ++p){
        if (transformed_obs[b].id == predicted[p].id){
          // If the id matches, then it's the closest neighbor
          mu_x = predicted[p].x;
          mu_y = predicted[p].y;
          particles[i].sense_x.push_back(mu_x);
          particles[i].sense_y.push_back(mu_y);
        }
      }
      prob = multiv_prob(std_x, std_y, x_obs, y_obs, mu_x, mu_y);
      if (prob == 0){
        prob = 1.0E-20;
      }
      w *= prob;
    }
    particles[i].weight = w;
    weights.push_back(w);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  vector<Particle> resample_p;
  for (int i = 0; i < num_particles; ++i){
    int index = dist(gen);
    resample_p.push_back(particles[index]);
  }
  particles = resample_p;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
