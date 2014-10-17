		  Rhyming Dictionary - Win32 Version

The Rhyming Dictionary is built to run in a terminal window on
UNIX-like systems (*BSD, Linux, Solaris, etc.).  Therefore, win32
users should expect it to look and feel different from native win32
applications.  This guide is intended to illustrate those differences.


Installation:

The win32 version will always be packaged in a zip archive which can
be unpacked anywhere on the system.  The only caveat is that the GDBM
database files (words.db, rhymes.db and multiple.db) must reside in
the same directory as the rhyme.exe executable.  Additionally, the
Cygwin .dll files must also reside in that directory so that rhyme.exe
can find them.


Usage:

You will need to open an MS-DOS prompt to use the Rhyming Dictionary;
it has no GUI and I have no immediate plans to build one.  Then,
simply enter the directory containing the rhyme.exe executable and
simply type:

rhyme word

to get the rhymes for that particular word.  Be aware, however, that
the Rhyming Dictionary expects your terminal window (in this case, the
MS-DOS prompt) to be resizable and have the capability of scrolling
back.  Therefore, it has no built in pager and rhymes may scroll off
the screen if you choose a word with a large number of rhymes.

To remedy this, you may want to increase the number of lines your
MS-DOS window displays, use a different DOS window other than the
default, or pipe the output through the "more" pager prior to display.
To perform the latter, type:

rhyme word | more

and all the rhymes will display one screen at a time.  Additionally,
to get a syllable count of a particular word, type:

rhyme -s word

and it will list only the number of syllables in that word.  Finally,
type:

rhyme -h

to get a full list of features.  The most interesting of these is the
interactive mode which allows you to get the rhymes for many words
without having to re-run rhyme.exe each time.  This can be entered by
typing:

rhyme -i

which will result in a "RHYME>" prompt.  From there you can type
words, one per line, and it will display the rhymes for each until you
enter a blank line.  Unfortunately, interactive mode doesn't have a
pager either and you will not be able to pipe the output to "more" as
in non-interactive mode.


License:

Both the Rhyming Dictionary and the Cygwin library .dll files it uses
for the win32 port are licensed under terms of the GPL.  See the
"COPYING" file for the full details of this license.  The source code
for the Rhyming Dictionary should be available at the same place you
found the win32 version, or it can be downloaded from:

http://rhyme.sourceforge.net

The Cygwin source code is available at:

http://www.cygwin.com


Contact:

If you have problems with either the win32 port of the Rhyming
Dictionary or its native UNIX version, I can be contacted at:

bri@cbs.umn.edu
