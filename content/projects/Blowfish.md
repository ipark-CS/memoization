+++
author = "ipark"
title = "The Blowfish Block Cipher System"
draft =  false
type = "projects"
layout = "projects"
description = ""
tags = ["project", "cryptograph", "blowfish", "block cipher",
]
+++
#### Abstract:

<blockquote>
<p>The Blowfish is one of block cipher systems. It was developed as a better substitute for the Triple-DES system.
Blowfish has following features:</p>
<ul>
<li>a. an input key length is a variable from minimum 32-bit up to maximum 448-bit;</li>
<li>large subkey tables to make a brute-force attack harder by using hexadecimal string of π value to initialize the 32-bit subkey set (P-array[18] and Sboxes[4][256]); and</li>
<li>one block encryption algorithm is used using fixed-sized all-zero strings in the subkey generation phase to make key-dependent subkeys.</li>
</ul>
<p>Overall, the Blowfish has many common features with other block ciphers as following, thus I could reuse many methods/classes from the cryptoUtil.</p>
<ul>
<li>large S-boxes data structure for lookup;</li>
<li>combined operations, .e.g. XOR mod 2^32; and</li>
<li>multiple rounds of Feistel iteration by swapping left and right half of a block</li>
</ul>
</blockquote>
<img src="/img/Blowfish/bf-1.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-2.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-3.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-4.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-5.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-7.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-9.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-8.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-10.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf11.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-12.png" width="570" style="border:1px solid black;">
<img src="/img/Blowfish/bf-13.png" width="570" style="border:1px solid black;">
