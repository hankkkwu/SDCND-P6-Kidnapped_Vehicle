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
  num_particles = 100;  // TODO: Set the number of particles
  // std::default_random_engine gen;   // generates pseudo-random numbers

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

  // std::default_random_engine gen;   // generates pseudo-random numbers

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // Create normal distribution (noise) for x, y and theta
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  for (int i = 0; i < num_particles; ++i){
    if (fabs(yaw_rate) > 0.0001){
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta+yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta+yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    // If yaw_rate < 0.0001, the car probably moving straight
    else{
      particles[i].x += velocity * cos(particles[i].theta) * delta_t;
      particles[i].y += velocity * sin(particles[i].theta) * delta_t;
    }
    // Add random Gaussian noise to x, y and theta
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
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
  int num_obs = observations.size();
  int num_pre = predicted.size();
  for (int i = 0; i < num_obs; ++i){
    // loop through each observation
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;
    double smallest_dist = std::numeric_limits<double>::max();
    for (int j = 0; j < num_pre; ++j){
       // loop through each predicted measurement
      double pre_x = predicted[j].x;
      double pre_y = predicted[j].y;
      double distance = dist(obs_x, obs_y, pre_x, pre_y);
      // Pick the smallest distance
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
  int num_obs = observations.size();
  int num_map = map_landmarks.landmark_list.size();

  for (int i = 0; i < num_particles; ++i){
    // Loop through each particle
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    vector<LandmarkObs> predicted;
    vector<LandmarkObs> transformed_obs;

    for (int j = 0; j < num_obs; ++j){
      // transform each observation marker from the vehicle's coordinates to the map's coordinates
      LandmarkObs tobs;
      tobs.x = cos(theta) * observations[j].x - sin(theta) * observations[j].y + x;
      tobs.y = sin(theta) * observations[j].x + cos(theta) * observations[j].y + y;
      transformed_obs.push_back(tobs);
    }

    for (int m = 0; m < num_map; ++m){
      // Loop through each landmark on the map to find the appropriate landmarks
      double map_x = map_landmarks.landmark_list[m].x_f;
      double map_y = map_landmarks.landmark_list[m].y_f;
      double distance = dist(x, y, map_x, map_y);
      // Pick landmarks that within the range of sensor's measurements
      if (distance < sensor_range){
        LandmarkObs predict_obs;
        predict_obs.x = map_x;
        predict_obs.y = map_y;
        predict_obs.id = map_landmarks.landmark_list[m].id_i;
        predicted.push_back(predict_obs);
      }
    }

    dataAssociation(predicted, transformed_obs);

    vector<int> association;
    vector<double> s_x;
    vector<double> s_y;
    int num_tobs = transformed_obs.size();
    particles[i].weight = 1.0;   // Set weight to 1.0 for multiplying
    for (int t = 0; t < num_tobs; ++t){
      double x_obs = transformed_obs[t].x;
      double y_obs = transformed_obs[t].y;
      // Using the id in transformed_obs to get the corresponding landmarks x, y position in map
      double mu_x = map_landmarks.landmark_list.at(transformed_obs[t].id-1).x_f;   // Id in map data is starting from 1 (map[0].id = 1)
      double mu_y = map_landmarks.landmark_list.at(transformed_obs[t].id-1).y_f;

      association.push_back(transformed_obs[t].id);
      s_x.push_back(x_obs);
      s_y.push_back(y_obs);

      // Calculate the weight for aech transformed observation
      double weight = multiv_prob(std_x, std_y, x_obs, y_obs, mu_x, mu_y);
      if (weight == 0){
        weight = 1.0E-5;
      }
      // Calculate the particle's final weight
      particles[i].weight *= weight;
    }
    weights.push_back(particles[i].weight);
    // For dispalying the blue lines in the simulator
    SetAssociations(particles[i], association, s_x, s_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Method 1: using std::discrete_distribution<int> dist()
  // std::default_random_engine gen;
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  vector<Particle> resample_p;

  for (int i = 0; i < num_particles; ++i){
    int index = dist(gen);
    resample_p.push_back(particles[index]);
  }
  particles = resample_p;
  // Clear the weights(vector) for the next step
  weights.clear();

  // Method 2: using resampling wheel
  /*
	std::default_random_engine gen;
	std::uniform_int_distribution<int> distInt(0, num_particles-1);
  std::uniform_real_distribution<double> disDouble(0, 1.0);

  vector<Particle> resample_p;
  int index = distInt(gen);
  double beta = 0.0;
  double max_weight = *std::max_element(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; ++i){
    beta += 2.0 * max_weight * disDouble(gen);
    while(beta > weights[index]){
      beta -= weights[index];
      index = (index+1) % num_particles;
    }
    resample_p.push_back(particles[index]);
  }
  particles = resample_p;
  // Clear the weights(vector) for the next step
  weights.clear();
  */
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
