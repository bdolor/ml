<?php
/**
 * Getting OER
 * Text Processing
 *  • Ignoring case
 *  • Ignoring punctuation, cleaning up whitespace
 *  • Ignoring stop words
 *  • Removing html entities, or markup
 *  • Reducing words to their stem using stemming algorithms
 *  • Looking at potential improvements through creating a vocabulary of
 * two-word pairs, or bigrams. Quantifying Vocabularies
 *  • TF-IDF
 * Training the Classifier
 * Prediction
 */
require_once __DIR__ . '/vendor/autoload.php';

use Phpml\Classification\SVC;
use Phpml\Classification\NaiveBayes;
use Phpml\Classification\KNearestNeighbors;
use Phpml\Metric\ClassificationReport;
use Phpml\Pipeline;
use Phpml\Preprocessing\Imputer;
use Phpml\Preprocessing\Imputer\Strategy\MedianStrategy;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\WordTokenizer;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\FeatureExtraction\StopWords\English;

/*
|--------------------------------------------------------------------------
| Training
|--------------------------------------------------------------------------
|
|
|
|
*/
$training_samples = [
	'Concepts of Biology: OpenStax Sciences Biology Published by OpenStax College, Concepts of Biology is designed for the single-semester introduction to biology course for non-science majors, which for many students is their only college-level science course. As such, this course represents an important opportunity for students to develop the necessary knowledge, tools, and skills to make informed decisions as they continue with their lives. Rather than being mired down with facts and vocabulary, the typical non-science major student needs information presented in a way that is easy to read and understand. Even more importantly, the content should be meaningful. Students do much better when they understand why biology is relevant to their everyday lives. For these reasons, Concepts of Biology is grounded on an evolutionary basis and includes exciting features that highlight careers in the biological sciences and everyday applications of the concepts at hand. We also strive to show the interconnectedness of topics within this extremely broad discipline. In order to meet the needs of today’s instructors and students, we maintain the overall organization and coverage found in most syllabi for this course. Instructors can customize Concepts of Biology, adapting it to the approach that works best in their classroom. Concepts of Biology also includes an innovative art program that incorporates critical thinking and clicker questions to help students understand—and apply—key concepts',
	'Forest Measurements: An Applied Approach Sciences Biology This is a forest measurements textbook written for field technicians. Silvicultural applications and illustrations are provided to demonstrate the relevance of the measurements. Special “technique tips” for each skill are intended to help increase data collection accuracy and confidence. These include how to avoid common pitfalls, effective short cuts, and essentials for recording field data correctly. The emphasis is on elementary skills; it is not intended to be a timber cruising guide.',
	'Exploring Movie Construction and Production Arts Art, Media and Design Exploring Movie Construction and Production contains eight chapters of the major areas of film construction and production. The discussion covers theme, genre, narrative structure, character portrayal, story, plot, directing style, cinematography, and editing. Important terminology is defined and types of analysis are discussed and demonstrated. An extended example of how a movie description reflects the setting, narrative structure, or directing style is used throughout the book to illustrate building blocks of each theme. This approach to film instruction and analysis has proved beneficial to increasing students’ learning, while enhancing the creativity and critical thinking of the student.',
	'Graphic Design and Print Production Fundamentals Arts Art, Media and Design This textbook -- written by a group of select experts with a focus on different aspects of the design process, from creation to production -- addresses the many steps of creating and then producing physical, printed, or other imaged products that people interact with on a daily basis. It covers the concept that, while most modern graphic design is created on computers using design software, the ideas and concepts don’t stay on the computer. The ideas need to be completed in the computer software, then progress to an imaging (traditionally referred to as printing) process. Keywords are highlighted throughout and summarized in a Glossary at the end of the book, and each chapter includes exercises and suggested readings.',
	'Line E: Electrical Fundamentals Competency E-4: Use Multimeters Trades Common Core Whether you choose to work in an electrical trade, a mechanical trade, or one of the construction trades, you will probably be faced with using and/or maintaining a variety of electrical measuring instruments. This Competency will introduce you to three basic meters for measuring voltage, current, and resistance. You must have a basic understanding of the purpose and operation of each type of meter before you attempt to use one. If you connect a meter incorrectly, you not only risk damaging the instrument, but more importantly, you or some innocent bystander could receive a serious electrical shock.The following list of lines and competencies was generated with the goal of creating an entry-level trades training resource, while still offering the flexibility for lines to be used as stand-alone books. E-1 Describe the Basic Principles of Electricity, E-2 Identify Common Circuit Components and Their Symbols, E-3 Explain Wiring Connections, E-4 Use Multimeters. Special thanks to CAPER-BC (https://caperbc.ca/) for creating the audio recording for each Competency.',
	'Line D: Organizational Skills Competency D-6: Plan Projects Trades Common Core Every job is different and may have special requirements. Anticipating these requirements and planning accordingly is vital to ensuring that you have the materials, tools, and time to complete the job. To do this effectively you will need to have a clear understanding of the overall job. You will need to know what materials are required and be able to record organized and accurate notes on the time and materials used when the job is complete. Planning ahead saves time and money and makes a job more profitable. Effectively managing time and resources, including materials, and keeping detailed notes is very important whether you are working for another company or on your own. It allows your company to be more competitive and also provides a good base for estimating the costs of similar jobs in the future. The following list of lines and competencies was generated with the goal of creating an entry-level trades training resource, while still offering the flexibility for lines to be used as stand-alone books. Line D – Organizational Skills. D-1 Solve Trades Mathematical Problems, D-2 Apply Science Concepts to Trades Applications, D-3 Read Drawings and Specifications, D-4 Use Codes, Regulations, and Standards, D-5 Use Manufacturer and Supplier Documentation, D-6 Plan Projects. Special thanks to CAPER-BC (https://caperbc.ca/) for creating the audio recording for each Competency.',
];

$transformers = [
	new TokenCountVectorizer( new WordTokenizer(), new English() ),
	new Imputer( NULL, new MedianStrategy() ),
	new TfIdfTransformer(),
];
$estimator    = new SVC();
//$estimator = new \Phpml\Classification\NaiveBayes();
//$estimator = new \Phpml\Classification\KNearestNeighbors();

$targets = [
	'Biology',
	'Biology',
	'Arts',
	'Arts',
	'Trades',
	'Trades',
];

$training_pipeline = new Pipeline( $transformers, $estimator );
$training_pipeline->train( $training_samples, $targets );

/*
|--------------------------------------------------------------------------
| Predicting
|--------------------------------------------------------------------------
|
|
|
|
*/
$new_samples  = [
	'Knowing Home: Braiding Indigenous Science with Western Science, Book 1 Sciences Biology Since Indigenous peoples have developed time-proven approaches to sustaining both community and environment, Elders and young people are concerned that this rich legacy of Indigenous Science with its wealth of environmental knowledge and the wisdom of previous generations could disappear if it is not respected, studied and understood by today\'s children and youth. A perspective where relationships between home place and all other beings that inhabit the earth is vitally important to all residents—both inheritors of ancient Indigenous Knowledge and wisdom, and newcomers who can experience the engagement, joy and promise of science instilled with a sense of place. This book takes a step forward toward preserving and actively using the knowledge, stories, and lessons for today and future generations, and with it a worldview that informs everyday attitudes toward the earth. Knowing Home: Braiding Indigenous Science with Western Science is far more than a set of research papers or curriculum studies. The project outputs include both, but they are incorporated into a theoretical structure that can provide the methodological basis for future efforts that attempt to develop culturally responsive Indigenous Science curricula in home places. It is not just one or two angels to organize, but multiple interwoven approaches and cases that give this project its exceptional importance. Thus, the project outputs have been organized into two books. Book 1 provides an overview of why traditional knowledge and wisdom should be included in the science curriculum, a window into the science and technologies of the Indigenous peoples who live in Northwestern North America, Indigenous worldview, culturally responsive teaching strategies and curriculum models, and evaluative techniques. It is intended that the rich examples and cases, combined with the resources listed in the appendices, will enable teachers and students to explore Indigenous Science examples in the classroom; and in addition, support the development of culturally appropriate curriculum projects.',
	'Understanding Media and Culture Arts Art Media and Design This book’s title tells its intent. It is written to help you understand media and culture. The media and culture are so much a part of our days that sometimes it is difficult to step back and appreciate and apprehend their great impact on our lives. The book’s title, and the book itself, begin with a focus squarely on media. Think of your typical day. If you are like many people, you wake to a digital alarm clock or perhaps your cell phone. Soon after waking, you likely have a routine that involves some media. Some people immediately check the cell phone for text messages. Others will turn on the computer and check Facebook, email, or websites. Some people read the newspaper. Others listen to music on an iPod or CD. Some people will turn on the television and watch a weather channel, cable news, or Sports Center. Heading to work or class, you may chat on a cell phone or listen to music. Your classes likely employ various types of media from course management software to PowerPoint presentations to DVDs to YouTube. You may return home and relax with video games, television, movies, more Facebook, or music. You connect with friends on campus and beyond with text messages or Facebook. And your day may end as you fall asleep to digital music. Media for most of us are entwined with almost every aspect of life and work. Understanding media will not only help you appreciate the role of media in your life but also help you be a more informed citizen, a more savvy consumer, and a more successful worker. Media influence all those aspects of life as well.',
	'Line D: Organizational Skills Competency D-3: Read Drawings and Specifications Trades Common Core Some of the most important documents used in the workplace are the technical drawings, diagrams, and schematics that specify how fabrication and construction tasks will be carried out, or describe the composition and assembly of equipment. One of the essential skills for anyone involved in a trade is the ability to correctly interpret drawings. If you are in a construction or fabrication industry, you will need to be able to examine a drawing, take information from it, and visualize the finished product. If you are in a service or maintenance industry, you will need to interpret exploded drawings in order to properly repair or assemble equipment. The following list of lines and competencies was generated with the goal of creating an entry-level trades training resource, while still offering the flexibility for lines to be used as stand-alone books. Line D – Organizational Skills. D-1 Solve Trades Mathematical Problems, D-2 Apply Science Concepts to Trades Applications, D-3 Read Drawings and Specifications, D-4 Use Codes, Regulations, and Standards, D-5 Use Manufacturer and Supplier Documentation, D-6 Plan Projects. Special thanks to CAPER-BC (https://caperbc.ca/) for creating the audio recording for each Competency.',
];
$actualLabels = [ 'Biology', 'Arts', 'Trades' ];

$predicted = $training_pipeline->predict( $new_samples );

/*
|--------------------------------------------------------------------------
| Reporting
|--------------------------------------------------------------------------
|
|
|
|
*/
$report = new ClassificationReport( $actualLabels, $predicted );
echo "<pre>";

echo "<h2>Precision</h2>";
print_r( $report->getPrecision() );

echo "<h2>Recall</h2>";
print_r( $report->getRecall() );

echo "<h2>F1 Score</h2>";
print_r( $report->getF1score() );

echo "<h2>Support</h2>";
print_r( $report->getSupport() );

echo "<h2>Average</h2>";
print_r( $report->getAverage() );
