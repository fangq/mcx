#!/usr/bin/perl

use CGI;
use strict;
use DBI;
use URI::Escape;
use JSON::PP;
use Digest::MD5 qw(md5_hex);

my ($DBName,$DBUser,$DBPass,%DBErr,$dbh,$sth,$html,$page,$jobid,$savetime,$dbname,$md5key,$callback,$jobhash);
my $q = new CGI;
my $req;
my %jobstatus=(0=>'queued',1=>'initiated',2=>'created',3=>'running',4=>'completed',5=>'deleted',6=>'failed',7=>'invalid',8=>'cancelled', 9=>'writing output');

if($q->param('json') =~/"RNGSeed"/){
	$req=decode_json($q->param( 'json' ));
}

$DBName="dbi:SQLite:dbname=db/mcxcloud.db";
$DBUser="";
$DBPass="";
%DBErr=(RaiseError=>0,PrintError=>1);
$savetime=time();
$jobid=uc(join "", map { unpack "H*", chr(rand(256)) } 1..20);
$callback='addlog';
if(&V("license") eq ''){
  $dbname="mcxcloud";
}else{
  $dbname="mcxpub";
}

#print $q -> header(
#-type => 'application/javascript',
#-access_control_allow_origin => '*',
#-access_control_allow_headers => 'content-type,X-Requested-With',
#-access_control_allow_methods => 'GET,POST,OPTIONS',
#-access_control_allow_credentials => 'true',
#);

print "Content-Type: application/javascript\n\n";

if(&V("callback") ne ''){
    $callback=&V("callback");
};

if(&V("hash") ne '' && &V("id") ne ''){
    my $ct=&V("id");
    my $key=&V("hash");
    $dbname='mcxpub';
    $dbh=DBI->connect($DBName,$DBUser,$DBPass,\%DBErr) or die($DBI::errstr);
    $sth=$dbh->selectall_arrayref("select json from $dbname where time=$ct and hash='$key';");
    my $jsondata='';
    if(defined $sth->[0]){
        foreach my $rec (@{$sth}){
            ($jsondata)=@$rec;
        }
    }
    my %response=('status'=>"success",'hash'=>$key, 'json'=>$jsondata);
    $html =$callback.'('.JSON::PP->new->utf8->encode(\%response).")\n";
    $dbh->disconnect() or die($DBI::errstr);
}elsif(&V("limit") ne ''){
    my $offset=&V("offset");
    $dbname='mcxpub';
    $dbh=DBI->connect($DBName,$DBUser,$DBPass,\%DBErr) or die($DBI::errstr);
    if(&V("keyword") eq ''){
        $sth=$dbh->selectall_arrayref("select time,hash,title,comment,license,thumbnail,readcount from $dbname order by readcount desc limit ".&V("limit")." offset $offset;");
    }else{
        my $k=&V("keyword");
        $sth=$dbh->selectall_arrayref("select time,hash,title,comment,license,thumbnail,readcount from $dbname where title like '%$k%' or comment like '%$k%' limit ".&V("limit")." offset $offset;");
    }
    my @res=();
    if(defined $sth->[0]){
        foreach my $rec (@{$sth}){
            my ($id,$key,$title,$comment,$lic,$thumb)=@$rec;
            my %simu=('time'=>$id,'hash'=>$key,'title'=>$title,'comment'=>$comment,'license'=>$lic,'thumbnail'=>$thumb);
            push( @res, \%simu );
        }
    }
    $html =$callback.'('.JSON::PP->new->utf8->encode(\@res).")\n";
    $dbh->disconnect() or die($DBI::errstr);
}elsif($dbname eq 'mcxpub'){
    $dbh=DBI->connect($DBName,$DBUser,$DBPass,\%DBErr) or die($DBI::errstr);
    $md5key=md5_hex(&V("json"));
    $sth=$dbh->prepare("insert into $dbname (time,title,comment,license,name,inst,email,netname,json,thumbnail,hash,createtime) values (?,?,?,?,?,?,?,?,?,?,?,?)");
    $sth->execute($savetime,&V("title"),&V("comment"),&V("license"),&V("name"),&V("inst"),&V("email"),&V("netname"),&V("json"),&V("thumbnail"),$md5key,$savetime);
    $html =$callback.'({"status":"success","createtime":"'.$savetime.'","hash":"'.$md5key.'","dberror":"'.$DBI::errstr.'"})'."\n";
    $dbh->disconnect() or die($DBI::errstr);
}elsif(&V("email") ne '' && &V("json") ne ''){
    $dbh=DBI->connect($DBName,$DBUser,$DBPass,\%DBErr) or die($DBI::errstr);
    $md5key=md5_hex(&V("json"));
    $sth=$dbh->prepare("insert into $dbname (time,name,inst,email,netname,json,jobid,hash,status,priority) values (?,?,?,?,?,?,?,?,?,?)");
    $sth->execute($savetime,&V("name"),&V("inst"),&V("email"),&V("netname"),&V("json"),$jobid,$md5key,1,50);
    $html =$callback.'({"status":"success","jobid":"'.$jobid.'","hash":"'.$md5key.'","dberror":"'.$DBI::errstr.'"})'."\n";
    $dbh->disconnect() or die($DBI::errstr);
}elsif(&V("action") eq 'cancel'){
    $jobid=&V("jobid");
    $dbh=DBI->connect($DBName,$DBUser,$DBPass,\%DBErr) or die($DBI::errstr);
    $sth=$dbh->prepare("update $dbname set status=8 where jobid='$jobid';")
                 or die "Couldn't prepare statement: ";
    $sth->execute();
    $dbh->disconnect();
    $html =$callback.'({"status":"cancelled","jobid":"'.$jobid.'","dberror":"'.$DBI::errstr.'"})'."\n";
}else{
    my ($status,$output,$log);
    $status=7;
    $jobid=&V("jobid");
    $jobhash="_".&V("hash");
    if($jobid ne '' && not (-d "workspace/$jobid") && $jobhash ne '' && (-d "workspace/$jobhash") ){
        $jobid=$jobhash;
    }
    if(-e "workspace/$jobid/output.jnii" && not -z "workspace/$jobid/output.jnii"){
        $status=9;
        if(-e "workspace/$jobid/done"){
          open FF, "<workspace/$jobid/output.jnii" || die("can not open log file");
          chomp(my @lines = <FF>);
          close(FF);
          my $outstr=join(/\n/,@lines);
          my $logstr="";
          if(-e "workspace/$jobid/output.log" && not -z "workspace/$jobid/output.log"){
              open FF, "<workspace/$jobid/output.log" || die("can not open log file");
              chomp(my @logs = <FF>);
              close(FF);
              $logstr=join("\n",@logs);
          }
          $status=4;
          my %response=('status'=>$jobstatus{$status}, 'jobid'=>$jobid, 'output'=>$outstr, 'log'=>$logstr);
          $html =$callback.'('.JSON::PP->new->utf8->encode(\%response).")\n";
        }else{
          $status=3;
          my %response=('status'=>$jobstatus{$status}, 'jobid'=>$jobid);
          $html =$callback.'('.JSON::PP->new->utf8->encode(\%response).")\n";
        }
    }elsif(-e "workspace/$jobid/input.json" && not -z "workspace/$jobid/input.json"){
        $status=2;
        $html =$callback.'({"status":"'.$jobstatus{$status}.'","jobid":"'.$jobid.'"})'."\n";
    }elsif($jobid ne '' && -d "workspace/$jobid"){
        $status=1;
        $html =$callback.'({"status":"'.$jobstatus{$status}.'","jobid":"'.$jobid.'"})'."\n";
    }else{
        my $cmd="select status from $dbname where jobid='$jobid';";
        $dbh=DBI->connect($DBName,$DBUser,$DBPass,\%DBErr) or die($DBI::errstr);
        $sth=$dbh->selectall_arrayref("select status from $dbname where jobid='$jobid';");
        if(defined $sth->[0]){
           $status=join(//,@{$sth->[0]});
        }
        $html =$callback.'({"status":"'.$jobstatus{$status}.'","jobid":"'.$jobid.'"})'."\n";
        $dbh->disconnect() or die($DBI::errstr);
    }
}

print $html;

sub V{
    my ($id)=@_;
    my $val=$q->param($id);
    $val=$req->{$id} if($val eq '');
    $val=~ s/\+/ /g;
    return uri_unescape($val);
}
