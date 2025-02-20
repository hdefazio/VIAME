/*ckwg +29
 * Copyright 2017-2021 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Implementation of detected object set csv output
 */

#include "write_object_track_set_viame_csv.h"

#include "notes_to_attributes.h"

#include <ctime>
#include <sstream>
#include <iomanip>

namespace viame {


// -------------------------------------------------------------------------------
class write_object_track_set_viame_csv::priv
{
public:
  priv( write_object_track_set_viame_csv* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "write_object_track_set_viame_csv" ) )
    , m_first( true )
    , m_delim( "," )
    , m_stream_identifier( "" )
    , m_model_identifier( "" )
    , m_version_identifier( "" )
    , m_active_writing( false )
    , m_write_time_as_uid( false )
    , m_tot_option( "weighted_average" )
    , m_tot_ignore_class( "" )
    , m_frame_id_adjustment( 0 )
  { }

  ~priv() { }

  write_object_track_set_viame_csv* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  bool m_first;
  std::string m_delim;
  std::string m_stream_identifier;
  std::string m_model_identifier;
  std::string m_version_identifier;
  std::map< unsigned, kwiver::vital::track_sptr > m_tracks;
  bool m_active_writing;
  bool m_write_time_as_uid;

  std::string m_tot_option;
  std::string m_tot_ignore_class;
  int m_frame_id_adjustment;
  std::map< unsigned, std::string > m_frame_uids;

  std::string format_image_id( const kwiver::vital::object_track_state* ts );
};

std::string
write_object_track_set_viame_csv::priv
::format_image_id( const kwiver::vital::object_track_state* ts )
{
  if( m_write_time_as_uid )
  {
    char output[10];
    time_t rawtime = ts->time() / 1e6;
    unsigned msec = ts->time() / 1e4;
    std::string msec_str = std::to_string( msec );
    while( msec_str.size() < 3 )
    {
      msec_str = "0" + msec_str;
    }
    struct tm* tmp = gmtime( &rawtime );
    strftime( output, sizeof( output ), "%H:%M:%S", tmp );
    return std::string( output ) + "." + msec_str + " UTC";
  }
  else if( !m_frame_uids.empty() )
  {
    std::string fileuid = m_frame_uids[ ts->frame() ];

    const size_t last_slash_idx = fileuid.find_last_of("\\/");
    if ( std::string::npos != last_slash_idx )
    {
      fileuid.erase( 0, last_slash_idx + 1 );
    }

    return fileuid;
  }
  else
  {
    return m_stream_identifier;
  }
}

kwiver::vital::detected_object_type_sptr
compute_average_tot( kwiver::vital::track_sptr trk_ptr,
                     bool weighted = false,
                     bool scale_by_conf = false,
                     std::string ignore_class = "" )
{
  if( !trk_ptr )
  {
    return kwiver::vital::detected_object_type_sptr();
  }

  std::vector< std::string > output_names;
  std::vector< double > output_scores;

  double weighted_mass = 0.0;
  double weighted_non_ignore_mass = 0.0;
  double weighted_ignore_mass = 0.0;

  std::map< std::string, double > class_sum;
  double ignore_sum = 0.0;
  double conf_sum = 0.0;
  unsigned conf_count = 0;

  for( auto ts_ptr : *trk_ptr )
  {
    kwiver::vital::object_track_state* ts =
      static_cast< kwiver::vital::object_track_state* >( ts_ptr.get() );

    if( !ts->detection() )
    {
      continue;
    }

    kwiver::vital::detected_object_type_sptr dot = ts->detection()->type();

    if( dot )
    {
      double weight = ( weighted ? ts->detection()->confidence() : 1.0 );

      if( scale_by_conf )
      {
        conf_sum += ts->detection()->confidence();
        conf_count += 1;
      }

      bool ignore = ( dot->class_names().size() == 1 &&
                      dot->class_names()[0] == ignore_class );

      if( ignore )
      {
        ignore_sum += ( dot->score( ignore_class ) * weight );
        weighted_ignore_mass += weight;
      }
      else
      {
        for( const auto name : dot->class_names() )
        {
          class_sum[ name ] += ( dot->score( name ) * weight );
        }
        weighted_non_ignore_mass += weight;
      }

      weighted_mass += weight;
    }
  }

  double prob_scale_factor = 1.0;

  if( scale_by_conf && conf_count > 0 )
  {
    prob_scale_factor = 0.1 + 0.9 * ( conf_sum / conf_count );
  }

  if( weighted_mass > 0.0 && weighted_ignore_mass == 0.0 )
  {
    prob_scale_factor /= weighted_mass;
  }
  else if( weighted_ignore_mass > 0.0 && weighted_non_ignore_mass > 0.0 )
  {
    prob_scale_factor /= weighted_non_ignore_mass;
  }
  else if( weighted_ignore_mass > 0.0 )
  {
    class_sum[ ignore_class ] = ignore_sum;
    prob_scale_factor /= weighted_ignore_mass;
  }

  for( auto itr : class_sum )
  {
    output_names.push_back( itr.first );
    output_scores.push_back( prob_scale_factor * itr.second );
  }

  if( output_names.empty() )
  {
    return kwiver::vital::detected_object_type_sptr();
  }

  return std::make_shared< kwiver::vital::detected_object_type >(
    output_names, output_scores );
}


// ===============================================================================
write_object_track_set_viame_csv
::write_object_track_set_viame_csv()
  : d( new write_object_track_set_viame_csv::priv( this ) )
{
}


write_object_track_set_viame_csv
::~write_object_track_set_viame_csv()
{
}


void write_object_track_set_viame_csv
::close()
{
  if( d->m_active_writing )
  {
    // No flushing required
    write_object_track_set::close();
    return;
  }

  for( auto trk_pair : d->m_tracks )
  {
    auto trk_ptr = trk_pair.second;

    const kwiver::vital::detected_object_type_sptr trk_average_tot =
          ( d->m_tot_option == "detection" ? kwiver::vital::detected_object_type_sptr()
            : compute_average_tot( trk_ptr,
                d->m_tot_option.find( "weighted" ) != std::string::npos,
                d->m_tot_option.find( "scaled_by_conf" ) != std::string::npos,
                d->m_tot_ignore_class ) );

    for( auto ts_ptr : *trk_ptr )
    {
      kwiver::vital::object_track_state* ts =
        dynamic_cast< kwiver::vital::object_track_state* >( ts_ptr.get() );

      if( !ts )
      {
        LOG_ERROR( d->m_logger, "Invalid timestamp " << trk_ptr->id()
                                                     << " " << trk_ptr->size() );
        continue;
      }

      kwiver::vital::detected_object_sptr det = ts->detection();
      const kwiver::vital::bounding_box_d empty_box =
        kwiver::vital::bounding_box_d( -1, -1, -1, -1 );
      kwiver::vital::bounding_box_d bbox = ( det ? det->bounding_box() : empty_box );
      auto confidence = ( det ? det->confidence() : 0 );
      int frame_id = ts->frame() + d->m_frame_id_adjustment;

      stream() << trk_ptr->id() << d->m_delim            // 1: track id
               << d->format_image_id( ts ) << d->m_delim // 2: video or image id
               << frame_id << d->m_delim                 // 3: frame number
               << bbox.min_x() << d->m_delim             // 4: TL-x
               << bbox.min_y() << d->m_delim             // 5: TL-y
               << bbox.max_x() << d->m_delim             // 6: BR-x
               << bbox.max_y() << d->m_delim             // 7: BR-y
               << confidence << d->m_delim               // 8: confidence
               << "0";                                   // 9: length

      if( det )
      {
        const kwiver::vital::detected_object_type_sptr dot =
          ( d->m_tot_option == "detection" ? det->type() : trk_average_tot );

        if( dot )
        {
          const auto name_list( dot->class_names() );
          for( auto name : name_list )
          {
            stream() << d->m_delim << name << d->m_delim << dot->score( name );
          }
        }

        if( !det->notes().empty() )
        {
          stream() << notes_to_attributes( det->notes(), d->m_delim );
        }

        stream() << std::endl;
      }
    }

    // Flush stream to prevent buffer issues
    stream().flush();
  }

  write_object_track_set::close();
}


// -------------------------------------------------------------------------------
void
write_object_track_set_viame_csv
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_delim =
    config->get_value< std::string >( "delimiter", d->m_delim );
  d->m_stream_identifier =
    config->get_value< std::string >( "stream_identifier", d->m_stream_identifier );
  d->m_model_identifier =
    config->get_value< std::string >( "model_identifier", d->m_model_identifier );
  d->m_version_identifier =
    config->get_value< std::string >( "version_identifier", d->m_version_identifier );
  d->m_active_writing =
    config->get_value< bool >( "active_writing", d->m_active_writing );
  d->m_write_time_as_uid =
    config->get_value< bool >( "write_time_as_uid", d->m_write_time_as_uid );
  d->m_tot_option =
    config->get_value< std::string> ( "tot_option", d->m_tot_option );
  d->m_tot_ignore_class =
    config->get_value< std::string >( "tot_ignore_class", d->m_tot_ignore_class );
  d->m_frame_id_adjustment =
    config->get_value< int >( "frame_id_adjustment", d->m_frame_id_adjustment );
}


// -------------------------------------------------------------------------------
bool
write_object_track_set_viame_csv
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------
void
write_object_track_set_viame_csv
::write_set( const kwiver::vital::object_track_set_sptr& set,
             const kwiver::vital::timestamp& ts,
             const std::string& file_id )
{
  if( d->m_first )
  {
    std::time_t rawtime;
    struct tm * timeinfo;

    time ( &rawtime );
    timeinfo = localtime ( &rawtime );
    char* cp =  asctime( timeinfo );
    cp[ strlen( cp )-1 ] = 0; // remove trailing newline
    const std::string atime( cp );

    // Write file header(s)
    stream() << "# 1: Detection or Track-id,"
             << "  2: Video or Image Identifier,"
             << "  3: Unique Frame Identifier,"
             << "  4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),"
             << "  8: Detection or Length Confidence,"
             << "  9: Target Length (0 or -1 if invalid),"
             << "  10-11+: Repeated Species, Confidence Pairs or Attributes"
             << std::endl;

    stream() << "# Written on: " << atime
             << "   by: write_object_track_set_viame_csv"
             << std::endl;

    if( !d->m_model_identifier.empty() )
    {
      stream() << "# Computed using model identifier: "
               << d->m_model_identifier
               << std::endl;
    }

    if( !d->m_version_identifier.empty() )
    {
      stream() << "# Computed using software version: "
               << d->m_version_identifier
               << std::endl;
    }

    d->m_first = false;
  }

  if( !file_id.empty() && ts.has_valid_frame() )
  {
    d->m_frame_uids[ ts.get_frame() ] = file_id;
  }

  if( !set )
  {
    return;
  }

  if( !d->m_active_writing )
  {
    for( auto trk : set->tracks() )
    {
      d->m_tracks[ trk->id() ] = trk;
    }
  }
  else
  {
    for( auto trk_ptr : set->tracks() )
    {
      if( !trk_ptr || trk_ptr->empty() )
      {
        LOG_ERROR( d->m_logger, "Received invalid track" );
        continue;
      }

      kwiver::vital::object_track_state* state =
        dynamic_cast< kwiver::vital::object_track_state* >( trk_ptr->back().get() );

      if( !state )
      {
        LOG_ERROR( d->m_logger, "Invalid track state for track "
                                << trk_ptr->id()
                                << " of length "
                                << trk_ptr->size() );
        continue;
      }

      if( state->frame() != ts.get_frame() )
      {
        // Last state is in the past, it was already written.
        continue;
      }

      kwiver::vital::detected_object_sptr det = state->detection();

      const kwiver::vital::bounding_box_d empty_box =
        kwiver::vital::bounding_box_d( -1, -1, -1, -1 );

      kwiver::vital::bounding_box_d bbox = ( det ? det->bounding_box() : empty_box );

      auto confidence = ( det ? det->confidence() : 0 );
      int frame_id = state->frame() + d->m_frame_id_adjustment;

      stream() << trk_ptr->id() << d->m_delim               // 1: track id
               << d->format_image_id( state ) << d->m_delim // 2: video or image id
               << frame_id << d->m_delim                    // 3: frame number
               << bbox.min_x() << d->m_delim                // 4: TL-x
               << bbox.min_y() << d->m_delim                // 5: TL-y
               << bbox.max_x() << d->m_delim                // 6: BR-x
               << bbox.max_y() << d->m_delim                // 7: BR-y
               << confidence << d->m_delim                  // 8: confidence
               << "0";                                      // 9: length

      if( det )
      {
        const kwiver::vital::detected_object_type_sptr dot =
          ( d->m_tot_option == "detection" ? det->type() :
            compute_average_tot( trk_ptr,
              d->m_tot_option.find( "weighted" ) != std::string::npos,
              d->m_tot_option.find( "scaled_by_conf" ) != std::string::npos,
              d->m_tot_ignore_class ) );

        if( dot )
        {
          const auto name_list( dot->class_names() );
          for( auto name : name_list )
          {
            stream() << d->m_delim << name << d->m_delim << dot->score( name );
          }
        }

        if( !det->keypoints().empty() )
        {
          for( const auto& kp : det->keypoints() )
          {
            stream() << d->m_delim << "(kp) " << kp.first;
            stream() << " " << kp.second.value()[0] << " " << kp.second.value()[1];
          }
        }

        if( !det->notes().empty() )
        {
          stream() << notes_to_attributes( det->notes(), d->m_delim );
        }

        stream() << std::endl;
      }
    }

    // Flush stream to prevent buffer issues
    stream().flush();
  }
}

} // end namespace
