The PhysioNet/Computing in Cardiology Challenge 2017 (http://physionet.org/challenge/2017/)
April, 2017 (Copyright (C) 2017  Morteza Zabihi & Ali Bahrami Rad)


Installing:
---------------------------------------------------------------------------------------------------------
1. Make sure all the files are saved in a same folder.


Content Map:
---------------------------------------------------------------------------------------------------------
1. challenge *
	1.1. installing_needed_toolbox *
	1.2. feature_extraction_BLACKSWAN
			1.2.1. beat_beat_BLACKSWAN *
			1.2.2. HRV_BLACKSWAN *
			1.2.3. faeture_extraction_ver1_ind1_BLACKSWAN *
                        a. probability_BLACKSWAN *
                        b. entropy_BLACKSWAN *
			1.2.4. Pwave_BLACKSWAN *
			1.2.5. Twave__BLACKSWAN *
			1.2.6. waveletECG1_BLACKSWAN *
                        a. statECG1_BLACKSWAN *
                        b. morphoECG1_BLACKSWAN *
			1.2.7. comput_AFEv (Copyright (c) 2013, University of Oxford, Gari D Clifford,  Roberta Colloca, Julien Oster) Modified 
                        a. comp_dRR (Copyright (c) 2013, University of Oxford, Gari D Clifford,  Roberta Colloca, Julien Oster)
                        b. metrics  (Copyright (c) 2013, University of Oxford, Gari D Clifford,  Roberta Colloca, Julien Oster)
                        c. BPcount  (Copyright (c) 2013, University of Oxford, Gari D Clifford,  Roberta Colloca, Julien Oster)
			1.2.8. frequency_feature_BLACKSWAN *
			1.2.9. HRV_PS_BLACKSWAN *
                        a. phasespace  (Copyright (C) Alexandros Leontitsis)
                        b. fit_ellipse (Copyright (C) Ohad Gal) Modified
			1.3.0. PCA_FEATURE_BLACKSWAN *
			1.3.1. qrs_detect2 (Copyright (C) 2014  Joachim Behar)
			1.3.2. Homogeneity_ver1_BLACKSWAN *
			1.3.3. EX_Feature_BLACKSWAN *
						a. Equalization_BLACKSWAN *
						b. CheckQuality_BLACKSWAN *
			1.3.4. SegmentBased_Detection_BLACKSWAN *
						a. WaveBased_BLACKSWAN *
						b. converttreebagerlabel_BLACKSWAN *
			1.3.5. BEAT_BEAT_TIMEFEATURES_ver01_BLACKSWAN *
						a. Frequency_Beat_TIMEFEATURES_ver01_BLACKSWAN *
						b. beat_beat_BLACKSWAN_ver2 *
			1.3.6. NEWPS_BLACKSWAN *
                        a. InterX (Copyright (C) NS)
			1.3.7. BEAT_BEAT_TIMEFEATURES_ver5_BLACKSWAN *
			1.3.8. generateValidationSet (by Chengyu Liu and Qiao Li January 20 2017) Modified
			1.3.9. beads (Copyright (C) Laurent Duval) 
2. classification_BLACKSWAN *
3. convert_label_BLACKSWAN *
4. LICENSE
5. license_fit_ellipse
6. license_beads
7. readme *
8. dependencies
9. AUTHORS
10. Model *
11. Model_Feature *

*(Copyright (C) 2017  Morteza Zabihi & Ali Bahrami Rad)

License:
---------------------------------------------------------------------------------------------------------
The codes shown with "*" are released under the GNU General Public License.
You should have received a copy of the GNU General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.
By accessing the codes through Physionet webpage and/or by installing, copying, or otherwise
using this software, you agree to be bounded by the terms of GNU General Public License.
If you do not agree, do not install copy or use the software.
We make no representation about the suitability of this licensed deliverable for any purppose.
It is provided "as is" without express or implied warranty of any kind.
Any use of the licensed deliverables must include the above disclaimer.
In addition to the codes and files shown with "*", which are implemented by the authors, the following toolboxes are also used:

    1. comp_dRR				---->  by Gari D Clifford - Roberta Colloca - Julien Oster
    2. comput_AFEv			---->  by Gari D Clifford - Roberta Colloca - Julien Oster
    3. metrics				---->  by Gari D Clifford - Roberta Colloca - Julien Oster
    4. BPcount				---->  by Gari D Clifford - Roberta Colloca - Julien Oster
    5. qrs_detect2			---->  by Joachim Behar Oxford university, Intelligent Patient Monitoring Group (joachim.behar@eng.ox.ac.uk)
    6. phasespace			---->  by: Alexandros Leontitsis
                                     Link: Updated code: http://www.mathworks.com/matlabcentral/fileexchange/1597-chaotic-systems-toolbox/content/phasespace.m
    7. fit_ellipse			---->  by Ohad Gal
                                     Link: Updated code: https://se.mathworks.com/matlabcentral/fileexchange/3215-fit-ellipse
    8. InterX				----> by: NS
                                    Link: https://se.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections?focused=5165138&tab=function
	9. beads				----> by: Laurent Duval
									Link: https://se.mathworks.com/matlabcentral/fileexchange/49974-beads--baseline-estimation-and-denoising-w--sparsity--chromatogram-signals-
									
Authors:
---------------------------------------------------------------------------------------------------------
Morteza Zabihi(morteza.zabihi@gmail.com) & Ali Bahrami Rad(abahramir@gmail.com)